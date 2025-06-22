import torch
import torch.nn as nn
import torch.optim as optim
import zipfile
from torch.utils.data import DataLoader, Dataset, TensorDataset
from PIL import Image
import pandas as pd
import random
import numpy as np
import json
import wandb
import argparse
import gzip
import os
import torch.nn.functional as F
import timm
from tqdm import tqdm

# Cache images in memory to avoid reloading them multiple times
image_cache = {}

def load_image(file_name, zip_ref, transform):
    # Check if the image is already cached
    cache_key = (file_name, str(transform))  # Use string representation of transform for caching
    if cache_key in image_cache:
        return image_cache[cache_key]
    # Load and transform the image
    with zip_ref.open(file_name) as file:
        image = Image.open(file).convert('RGB')
        if transform:
            image = transform(image)
        # Cache the transformed image
        image_cache[cache_key] = image
    return image

class PerceptDataset(Dataset):
    def __init__(self, zip_path, filenames, labels, transform):
        self.targets = []
        self.images = []
        self.file_names = []  # Initialize empty list for file names
        self.transform = transform
        self.zip_path = zip_path
        self.filenames = filenames
        self.labels = labels

        # Open the zip file containing images
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Get the list of file names in the specified subfolder
            all_file_names = zip_ref.namelist()
            # Load images and their corresponding labels
            for file_name in all_file_names:
                if file_name.endswith(".tif") and file_name.split("/")[-1] in filenames:
                    image = load_image(file_name, zip_ref, self.transform)
                    self.images.append(image)
                    if labels is not None:
                        # Get the index of the filename in the filenames list
                        idx = filenames.index(file_name.split("/")[-1])
                        self.targets.append(labels[idx])
                    self.file_names.append(file_name.split("/")[-1])

    def __len__(self):
        # Ensure the number of images matches the number of targets
        if self.labels is not None and len(self.images) != len(self.targets):
            raise AssertionError("The number of images does not match the number of targets.")
        return len(self.images)
    
    def __getitem__(self, idx):
        # Return the image and its corresponding target as a tensor (if labels are provided)
        if self.labels is not None:
            return self.images[idx], torch.tensor(self.targets[idx], dtype=torch.long), self.file_names[idx]

# Define worker init function outside the extract_features function
def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)

def extract_features(task_name, architecture, implant_type, model_type, 
                     train_filenames, train_labels, test_filenames, test_labels, 
                     zip_path, device, g, batch_size):
    """
    Extract features from a pre-trained model for all images
    
    Args:
        task_name: Name of the task
        architecture: Model architecture to use
        implant_type: Type of implant
        model_type: Type of model
        train_filenames: List of filenames for training
        test_filenames: List of filenames for testing
        zip_path: Path to the zip file containing images
        device: Device to use for feature extraction
        g: Random generator for reproducibility
        batch_size: Batch size for feature extraction
    
    Returns:
        Dictionary containing extracted features and labels for both train and test sets
    """
    print(f"Extracting features for {task_name} - {implant_type} - {model_type} - {architecture}")

    # Use timm's transforms and feature extraction
    # Create a feature extractor model (without classifier)
    feature_model = timm.create_model(architecture, pretrained=True, num_classes=0, global_pool='')
    # Get appropriate transforms
    data_config = timm.data.resolve_data_config({}, model=feature_model)
    transform = timm.data.transforms_factory.create_transform(**data_config)
    
    # Create datasets
    test_dataset = PerceptDataset(zip_path, test_filenames, test_labels, transform)
    train_dataset = PerceptDataset(zip_path, train_filenames, train_labels, transform)
    
    # Create data loaders
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        worker_init_fn=worker_init_fn,  # Use the global function instead of lambda
        generator=g,
        num_workers=4,
        pin_memory=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        worker_init_fn=worker_init_fn,  # Use the global function instead of lambda
        generator=g,
        num_workers=4,
        pin_memory=True,
        shuffle=False  # No need to shuffle for feature extraction
    )
    
    # Place model on device and set to evaluation mode
    feature_model.to(device)
    feature_model.eval()
    
    # Extract features for training set
    train_features = []
    train_targets = []
    train_filenames_list = []
    
    with torch.no_grad():
        for inputs, targets, filenames in tqdm(train_loader, desc="Extracting train features"):
            inputs = inputs.to(device)
            features = feature_model(inputs)
            # Flatten spatial dimensions if needed
            if len(features.shape) > 2:
                features = features.mean([2, 3]) if len(features.shape) == 4 else features.reshape(features.size(0), -1)
            train_features.append(features.cpu())
            train_targets.append(targets)
            train_filenames_list.extend(filenames)
    
    # Extract features for test set
    test_features = []
    test_targets = []
    test_filenames_list = []
    
    with torch.no_grad():
        for inputs, targets, filenames in tqdm(test_dataloader, desc="Extracting test features"):
            inputs = inputs.to(device)
            features = feature_model(inputs)
            # Flatten spatial dimensions if needed
            if len(features.shape) > 2:
                features = features.mean([2, 3]) if len(features.shape) == 4 else features.reshape(features.size(0), -1)
            test_features.append(features.cpu())
            test_targets.append(targets)
            test_filenames_list.extend(filenames)
    
    # Concatenate features
    train_features = torch.cat(train_features, dim=0)
    train_targets = torch.cat(train_targets, dim=0)
    
    test_features = torch.cat(test_features, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    # Create a dictionary with all the extracted data
    extracted_data = {
        'train_features': train_features,
        'train_targets': train_targets,
        'train_filenames': train_filenames_list,
        'test_features': test_features,
        'test_targets': test_targets,
        'test_filenames': test_filenames_list
    }
    
    return extracted_data

def train_linear_readout(task_name, architecture, run, implant_type, model_type, features_data, device, labels, num_epochs, batch_size):
    """
    Train a linear classifier on top of extracted features
    
    Args:
        task_name: Name of the task
        architecture: Model architecture used for features
        run: Run number for reproducibility
        implant_type: Type of implant
        model_type: Type of model
        features_data: Dictionary containing extracted features and labels
        device: Device to use for training
        labels: List of label names
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
    """
    run_name = f"{task_name}-{implant_type}-{model_type}-{architecture}-{run}"  
    preds_path = f"{task_name}/model_predictions/{implant_type}/{model_type}/{architecture}/{run}.json.gz"
    
    if os.path.exists(preds_path):
        print(f"Predictions already exist at {preds_path}, skipping training")
        return
    
    os.makedirs(os.path.dirname(preds_path), exist_ok=True)
    
    # Initialize a Weights & Biases run
    wandb.init(project="Noyce-Model-Zoo", name=run_name, config={
        "task_name": task_name,
        "architecture": architecture,
        "run": run,
        "implant_type": implant_type,
        "model_type": model_type,
    })
    
    # Get features and targets
    train_features = features_data['train_features'].to(device)
    train_targets = features_data['train_targets'].to(device)
    test_features = features_data['test_features'].to(device)
    test_targets = features_data['test_targets'].to(device)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_features, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Get number of input features and classes
    input_dim = train_features.shape[1]
    num_classes = len(set(train_targets.cpu().numpy()))
    
    # Create a linear model
    model = nn.Linear(input_dim, num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        })
    
    # Evaluate on test set
    model.eval()
    test_correct = 0
    test_total = 0
    all_targets, all_predictions, all_probabilities = [], [], []
    
    with torch.no_grad():
        outputs = model(test_features)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        test_correct += (predicted == test_targets).sum().item()
        test_total += test_targets.size(0)
        all_targets = test_targets.cpu().numpy()
        all_predictions = predicted.cpu().numpy()
        all_probabilities = probabilities.cpu().numpy()
    
    # Log test accuracy
    test_accuracy = 100 * test_correct / test_total
    wandb.log({
        "test_accuracy": test_accuracy,
        "test_size": test_total,
        "train_size": len(train_dataset)
    })
    
    # Create results dataframe
    results_df = pd.DataFrame({
        "filename": features_data['test_filenames'],
        "target": [labels[t] for t in all_targets],
        "prediction": [labels[p] for p in all_predictions],
        "probabilities": [list(prob) for prob in all_probabilities],
    })
    results_df["result"] = results_df["prediction"] == results_df["target"]
    
    # Save predictions
    with gzip.open(preds_path, "wt", newline='') as f:
        results_df.to_json(f, orient="records")
    
    wandb.finish()

def run_multiple_models(task_name, architecture, implant_type, model_type, 
                        train_filenames, train_labels, test_filenames, test_labels, 
                        zip_path, device, labels, num_runs, num_epochs, batch_size):
    """
    Run multiple model training with the same features
    
    Args:
        task_name: Name of the task
        architecture: Model architecture to use
        implant_type: Type of implant
        model_type: Type of model
        train_filenames: List of filenames for training
        train_labels: List of label names for training
        test_filenames: List of filenames for testing
        test_labels: List of label names for testing
        zip_path: Path to the zip file containing images
        device: Device to use for training
        labels: List of label names
        num_runs: Number of runs to perform
    """
    # Check if all prediction files already exist
    all_predictions_exist = True
    for run in range(1, num_runs + 1):
        preds_path = f"{task_name}/model_predictions/{implant_type}/{model_type}/{architecture}/{run}.json.gz"
        if not os.path.exists(preds_path):
            all_predictions_exist = False
            break
    
    if all_predictions_exist:
        print(f"All predictions already exist for {task_name}-{implant_type}-{model_type}-{architecture}, skipping...")
        return
    
    # Extract features once for all runs
    features_data = extract_features(
        task_name=task_name,
        architecture=architecture,
        implant_type=implant_type,
        model_type=model_type,
        train_filenames=train_filenames,
        train_labels=train_labels,
        test_filenames=test_filenames,
        test_labels=test_labels,
        zip_path=zip_path,
        device=device,
        g=torch.Generator().manual_seed(42),  # Fixed seed for feature extraction
        batch_size=batch_size
    )
    
    # Train multiple linear models with different seeds
    for run in range(1, num_runs + 1):
        print(f"Running {task_name}-{implant_type}-{model_type}-{architecture}, run {run}/{num_runs}")
        
        # Set random seeds for reproducibility
        random.seed(run)
        np.random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed(run)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        g = torch.Generator()
        g.manual_seed(run)
        
        # Train linear model
        train_linear_readout(
            task_name=task_name,
            architecture=architecture,
            run=run,
            implant_type=implant_type,
            model_type=model_type,
            features_data=features_data,
            device=device,
            labels=labels,
            num_epochs=num_epochs,
            batch_size=batch_size
        )

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process experiment parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--implant_type", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--num_runs", type=int, default=12)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)
    with open(f"../data/{args.task}/train_test.json", "r") as f:
        data_split = json.load(f)
    # Get train and test file information directly from JSON
    train_filenames = data_split[args.implant_type][args.model_type]["train_filenames"]
    train_labels = data_split[args.implant_type][args.model_type]["train_labels"]
    test_filenames = data_split[args.implant_type][args.model_type]["test_filenames"]
    test_labels = data_split[args.implant_type][args.model_type]["test_labels"]
    
    # Create label-to-index mapping from config
    label_to_idx = {label: idx for idx, label in enumerate(config["tasks"][args.task]["labels"])}
    # Convert test shape types to indices
    train_labels = [label_to_idx[label] for label in train_labels]
    test_labels = [label_to_idx[label] for label in test_labels]

    zip_path = f"../data/{args.task}/percepts.zip"
    
    # Sanity checks
    train_filenames_set = set(train_filenames)
    test_filenames_set = set(test_filenames)
    overlap = train_filenames_set.intersection(test_filenames_set)
    assert len(overlap) == 0, f"Data leakage detected: {len(overlap)} overlapping percept filenames between training and test sets."

    # Run multiple models
    run_multiple_models(
        task_name=args.task,
        architecture=args.architecture,
        implant_type=args.implant_type,
        model_type=args.model_type,
        train_filenames=train_filenames,
        train_labels=train_labels,
        test_filenames=test_filenames,
        test_labels=test_labels,
        zip_path=zip_path,
        device=device,
        labels=config["tasks"][args.task]["labels"],
        num_runs=args.num_runs,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )
