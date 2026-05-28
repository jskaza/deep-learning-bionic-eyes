"""
Train image classification models on bionic eye percept data.

Three training modes are supported (--mode):

  ridge_readout:    Frozen backbone → Ridge classifier (sklearn)
                    - Single deterministic run, no gradient descent
                    - Alpha values are tuned via cross-validation

  fine_tune_head:   Frozen backbone → nn.Linear trained with AdamW
                    - Multiple runs with different seeds
                    - Same features as ridge, but learned via SGD

  fine_tune_all:    Full model trained end-to-end with AdamW
                    - Backbone (low LR) + head (high LR)
                    - Multiple runs with different seeds
"""
import argparse
import gzip
import json
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifierCV
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# =============================================================================
# DATA LOADING
# =============================================================================

image_cache = {}  # Cache images in memory to avoid reloading


def load_image(file_name, zip_ref, transform):
    # Check if the image is already cached
    cache_key = (file_name, str(transform))  # Use string representation of transform for caching
    if cache_key in image_cache:
        return image_cache[cache_key]
    # Load and transform the image
    with zip_ref.open(file_name) as file:
        image = Image.open(file).convert("RGB")
        if transform:
            image = transform(image)
        # Cache the transformed image
        image_cache[cache_key] = image
    return image


class PerceptDataset(Dataset):
    def __init__(self, zip_path, filenames, labels, transform):
        self.targets = []
        self.images = []
        self.file_names = []
        self.transform = transform
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
        if self.labels is None:
            return self.images[idx], self.file_names[idx]
        return self.images[idx], torch.tensor(self.targets[idx], dtype=torch.long), self.file_names[idx]


def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)


# =============================================================================
# FEATURE EXTRACTION (shared by ridge_readout and fine_tune_head)
# =============================================================================


def extract_features(
    task_name,
    architecture,
    architecture_pretty_name,
    implant_type,
    model_type,
    train_filenames,
    train_labels,
    test_filenames,
    test_labels,
    zip_path,
    device,
    g,
    batch_size,
):
    """Extract features from frozen pretrained backbone for train/test sets."""
    print(f"Extracting features for {task_name} - {implant_type} - {model_type} - {architecture_pretty_name}")

    # Create a feature extractor model (without classifier)
    feature_model = timm.create_model(architecture, pretrained=True, num_classes=0, global_pool="")
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
        worker_init_fn=worker_init_fn,
        generator=g,
        num_workers=4,
        pin_memory=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        worker_init_fn=worker_init_fn,
        generator=g,
        num_workers=4,
        pin_memory=True,
        shuffle=False,  # No need to shuffle for feature extraction
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
                features = (
                    features.mean([2, 3]) if len(features.shape) == 4 else features.reshape(features.size(0), -1)
                )
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
                features = (
                    features.mean([2, 3]) if len(features.shape) == 4 else features.reshape(features.size(0), -1)
                )
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
        "train_features": train_features,
        "train_targets": train_targets,
        "train_filenames": train_filenames_list,
        "test_features": test_features,
        "test_targets": test_targets,
        "test_filenames": test_filenames_list,
    }

    return extracted_data


# =============================================================================
# MODE: ridge_readout
# =============================================================================


def train_ridge_readout(
    task_name,
    architecture_pretty_name,
    implant_type,
    model_type,
    features_data,
    labels,
):
    """Fit RidgeClassifierCV on extracted features. Deterministic, no seeds needed."""
    # Get features and targets as numpy arrays
    train_features = features_data["train_features"].numpy()
    train_targets = features_data["train_targets"].numpy()
    test_features = features_data["test_features"].numpy()
    test_targets = features_data["test_targets"].numpy()

    # Get number of parameters (input_dim * num_classes + num_classes for bias)
    input_dim = train_features.shape[1]
    num_classes = len(labels)
    n_params = input_dim * num_classes + num_classes

    # Fit Ridge classifier with CV to find optimal alpha
    clf = RidgeClassifierCV(alphas=np.logspace(-6, 6, 50))
    clf.fit(train_features, train_targets)

    # Get predictions and decision function scores (analogous to logits)
    all_predictions = clf.predict(test_features)
    all_logits = clf.decision_function(test_features)
    
    # Handle binary classification case where decision_function returns 1D array
    if all_logits.ndim == 1:
        all_logits = np.column_stack([-all_logits, all_logits])
    
    all_probabilities = softmax(all_logits, axis=1)

    # Create results dataframe
    results_df = pd.DataFrame(
        {
            "filename": features_data["test_filenames"],
            "target": [labels[t] for t in test_targets],
            "prediction": [labels[p] for p in all_predictions],
            "logits": [list(logit) for logit in all_logits],
            "probabilities": [list(prob) for prob in all_probabilities],
        }
    )
    results_df["result"] = results_df["prediction"] == results_df["target"]
    preds_path = f"{task_name}/ridge_readout_predictions/{implant_type}/{model_type}/{architecture_pretty_name} ({n_params} params)/ridge_classifier.json.gz"
    os.makedirs(os.path.dirname(preds_path), exist_ok=True)
    # Save predictions
    with gzip.open(preds_path, "wt", newline="") as f:
        results_df.to_json(f, orient="records")

    return {"n_params": n_params}

def run_ridge_readout(
    task_name,
    architecture,
    architecture_pretty_name,
    implant_type,
    model_type,
    train_filenames,
    train_labels,
    test_filenames,
    test_labels,
    zip_path,
    device,
    labels,
    batch_size,
):
    """Entry point for ridge_readout mode. Single deterministic run."""
    print(f"Training ridge readout {task_name}-{implant_type}-{model_type}-{architecture_pretty_name}")

    # Extract features
    features_data = extract_features(
        task_name=task_name,
        architecture=architecture,
        architecture_pretty_name=architecture_pretty_name,
        implant_type=implant_type,
        model_type=model_type,
        train_filenames=train_filenames,
        train_labels=train_labels,
        test_filenames=test_filenames,
        test_labels=test_labels,
        zip_path=zip_path,
        device=device,
        g=torch.Generator().manual_seed(42),
        batch_size=batch_size,
    )

    # Train Ridge classifier
    train_ridge_readout(
        task_name=task_name,
        architecture_pretty_name=architecture_pretty_name,
        implant_type=implant_type,
        model_type=model_type,
        features_data=features_data,
        labels=labels,
    )


# =============================================================================
# MODE: fine_tune_all
# =============================================================================


def train_fine_tune_all(
    task_name,
    architecture,
    architecture_pretty_name,
    run,
    implant_type,
    model_type,
    train_filenames,
    train_labels,
    test_filenames,
    test_labels,
    zip_path,
    device,
    labels,
    num_epochs,
    batch_size,
    head_lr,
    backbone_lr,
    g,
):
    """Train full model end-to-end. Backbone uses backbone_lr, head uses head_lr."""

    # Create model with full classifier head
    num_classes = len(labels)
    model = timm.create_model(architecture, pretrained=True, num_classes=num_classes)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # Get appropriate transforms
    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.transforms_factory.create_transform(**data_config)

    # Create datasets
    train_dataset = PerceptDataset(zip_path, train_filenames, train_labels, transform)
    test_dataset = PerceptDataset(zip_path, test_filenames, test_labels, transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        generator=g,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        generator=g,
        num_workers=4,
        pin_memory=True,
    )

    # Separate parameters into head and backbone groups for different learning rates
    head_params = set(model.get_classifier().parameters())
    backbone_params = [p for p in model.parameters() if p not in head_params]
    head_params = list(head_params)


    # Define loss function and optimizer with parameter groups
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ])

    # Training loop
    loss_history = []
    for _ in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        loss_history.append(epoch_loss)

    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_logits = []
    test_filenames_list = []
    with torch.no_grad():
        for inputs, targets, filenames in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            logits = outputs.cpu().numpy()
            probabilities = softmax(logits, axis=1)
            all_logits.append(logits)
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(np.argmax(logits, axis=1))
            all_probabilities.append(probabilities)
            test_filenames_list.extend(filenames)

    # Flatten per-batch arrays into per-sample arrays
    all_logits = np.concatenate(all_logits, axis=0)
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    # Create results dataframe
    results_df = pd.DataFrame(
        {
            "filename": test_filenames_list,
            "target": [labels[int(t)] for t in all_targets],
            "prediction": [labels[int(p)] for p in all_predictions],
            "logits": [list(row) for row in all_logits],
            "probabilities": [list(row) for row in all_probabilities],
        }
    )
    results_df["result"] = results_df["prediction"] == results_df["target"]

    preds_path = f"{task_name}/full_model_predictions/{implant_type}/{model_type}/{architecture_pretty_name} ({n_params} params)/{run}.json.gz"

    os.makedirs(os.path.dirname(preds_path), exist_ok=True)
    # Save predictions
    with gzip.open(preds_path, "wt", newline="") as f:
        results_df.to_json(f, orient="records")

    return {"loss_history": loss_history, "n_params": n_params}


# =============================================================================
# MODE: fine_tune_head
# =============================================================================


def train_fine_tune_head(
    task_name,
    architecture_pretty_name,
    run,
    implant_type,
    model_type,
    features_data,
    device,
    labels,
    num_epochs,
    batch_size,
    lr,
    g,
):
    """Train nn.Linear on pre-extracted features with AdamW. Single run."""
    # Get features and targets
    train_features = features_data["train_features"]
    train_targets = features_data["train_targets"]
    test_features = features_data["test_features"]
    test_targets = features_data["test_targets"]

    # Get dimensions
    input_dim = train_features.shape[1]
    num_classes = len(labels)
    n_params = input_dim * num_classes + num_classes  # weights + bias

    # Create a simple linear model
    linear_head = nn.Linear(input_dim, num_classes)
    linear_head.to(device)

    # Create simple tensor datasets for features
    train_tensor_dataset = torch.utils.data.TensorDataset(train_features, train_targets)
    test_tensor_dataset = torch.utils.data.TensorDataset(
        test_features, test_targets, 
        torch.arange(len(test_features))  # indices to map back to filenames
    )

    train_loader = DataLoader(
        train_tensor_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )

    test_loader = DataLoader(
        test_tensor_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(linear_head.parameters(), lr=lr)

    # Training loop
    loss_history = []
    for _ in range(num_epochs):
        linear_head.train()
        running_loss = 0.0

        for features_batch, targets_batch in train_loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()
            outputs = linear_head(features_batch)
            loss = criterion(outputs, targets_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        loss_history.append(epoch_loss)

    # Evaluation
    linear_head.eval()
    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_logits = []
    all_indices = []

    with torch.no_grad():
        for features_batch, targets_batch, indices_batch in test_loader:
            features_batch = features_batch.to(device)
            outputs = linear_head(features_batch)
            logits = outputs.cpu().numpy()
            probabilities = softmax(logits, axis=1)
            all_logits.append(logits)
            all_targets.append(targets_batch.numpy())
            all_predictions.append(np.argmax(logits, axis=1))
            all_probabilities.append(probabilities)
            all_indices.extend(indices_batch.numpy())

    # Flatten per-batch arrays into per-sample arrays
    all_logits = np.concatenate(all_logits, axis=0)
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    # Get filenames in correct order
    test_filenames_list = [features_data["test_filenames"][i] for i in all_indices]

    # Create results dataframe
    results_df = pd.DataFrame(
        {
            "filename": test_filenames_list,
            "target": [labels[int(t)] for t in all_targets],
            "prediction": [labels[int(p)] for p in all_predictions],
            "logits": [list(row) for row in all_logits],
            "probabilities": [list(row) for row in all_probabilities],
        }
    )
    results_df["result"] = results_df["prediction"] == results_df["target"]

    preds_path = f"{task_name}/head_model_predictions/{implant_type}/{model_type}/{architecture_pretty_name} ({n_params} params)/{run}.json.gz"

    os.makedirs(os.path.dirname(preds_path), exist_ok=True)
    # Save predictions
    with gzip.open(preds_path, "wt", newline="") as f:
        results_df.to_json(f, orient="records")

    return {"loss_history": loss_history, "n_params": n_params}


def run_fine_tune_head(
    task_name,
    architecture,
    architecture_pretty_name,
    implant_type,
    model_type,
    train_filenames,
    train_labels,
    test_filenames,
    test_labels,
    zip_path,
    device,
    labels,
    num_runs,
    num_epochs,
    batch_size,
    lr
):
    """Entry point for fine_tune_head mode. Extracts features once, trains head num_runs times."""
    # Extract features once (deterministic)
    print(f"Extracting features for fine_tune_head: {task_name}-{implant_type}-{model_type}-{architecture_pretty_name}")
    features_data = extract_features(
        task_name=task_name,
        architecture=architecture,
        architecture_pretty_name=architecture_pretty_name,
        implant_type=implant_type,
        model_type=model_type,
        train_filenames=train_filenames,
        train_labels=train_labels,
        test_filenames=test_filenames,
        test_labels=test_labels,
        zip_path=zip_path,
        device=device,
        g=torch.Generator().manual_seed(42),
        batch_size=batch_size,
    )

    # Train multiple head models with different seeds
    all_loss_histories = []
    n_params = None
    for run in range(1, num_runs + 1):
        print(f"Training fine_tune_head {task_name}-{implant_type}-{model_type}-{architecture_pretty_name}, run {run}/{num_runs}")

        # Set random seeds for reproducibility
        random.seed(run)
        np.random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed(run)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        g = torch.Generator()
        g.manual_seed(run)

        # Train head model
        metrics = train_fine_tune_head(
            task_name=task_name,
            architecture_pretty_name=architecture_pretty_name,
            run=run,
            implant_type=implant_type,
            model_type=model_type,
            features_data=features_data,
            device=device,
            labels=labels,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            g=g,
        )
        all_loss_histories.append(metrics["loss_history"])
        if n_params is None:
            n_params = metrics["n_params"]

    # Plot training loss curves for all runs on a single figure
    if len(all_loss_histories) > 0 and n_params is not None:
        plt.figure()
        for idx, loss_history in enumerate(all_loss_histories, start=1):
            epochs = list(range(1, len(loss_history) + 1))
            plt.plot(epochs, loss_history, label=f"Run {idx}")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title(
            f"Head model training loss\n{task_name} - {implant_type} - {model_type} - {architecture_pretty_name}"
        )
        plt.legend()
        plt.tight_layout()
        loss_curve_path = f"{task_name}/head_model_loss_curves/{implant_type}/{model_type}/{architecture_pretty_name} ({n_params} params)/training_loss.png"
        os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
        plt.savefig(loss_curve_path)
        plt.close()


def run_fine_tune_all(
    task_name,
    architecture,
    architecture_pretty_name,
    implant_type,
    model_type,
    train_filenames,
    train_labels,
    test_filenames,
    test_labels,
    zip_path,
    device,
    labels,
    num_runs,
    num_epochs,
    batch_size,
    head_lr,
    backbone_lr,
):
    """Entry point for fine_tune_all mode. Trains full model from scratch num_runs times."""
    # Train multiple models with different seeds
    all_loss_histories = []
    n_params = None
    for run in range(1, num_runs + 1):
        print(f"Training fine_tune_all {task_name}-{implant_type}-{model_type}-{architecture_pretty_name}, run {run}/{num_runs}")

        # Set random seeds for reproducibility
        random.seed(run)
        np.random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed(run)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        g = torch.Generator()
        g.manual_seed(run)

        # Train full model
        metrics = train_fine_tune_all(
            task_name=task_name,
            architecture=architecture,
            architecture_pretty_name=architecture_pretty_name,
            run=run,
            implant_type=implant_type,
            model_type=model_type,
            train_filenames=train_filenames,
            train_labels=train_labels,
            test_filenames=test_filenames,
            test_labels=test_labels,
            zip_path=zip_path,
            device=device,
            labels=labels,
            num_epochs=num_epochs,
            batch_size=batch_size,
            head_lr=head_lr,
            backbone_lr=backbone_lr,
            g=g,
        )
        all_loss_histories.append(metrics["loss_history"])
        if n_params is None:
            n_params = metrics["n_params"]

    # Plot training loss curves for all runs on a single figure
    if len(all_loss_histories) > 0 and n_params is not None:
        plt.figure()
        for idx, loss_history in enumerate(all_loss_histories, start=1):
            epochs = list(range(1, len(loss_history) + 1))
            plt.plot(epochs, loss_history, label=f"Run {idx}")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title(
            f"Full model training loss\n{task_name} - {implant_type} - {model_type} - {architecture_pretty_name}"
        )
        plt.legend()
        plt.tight_layout()
        loss_curve_path = f"{task_name}/full_model_loss_curves/{implant_type}/{model_type}/{architecture_pretty_name} ({n_params} params)/training_loss.png"
        os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
        plt.savefig(loss_curve_path)
        plt.close()


# =============================================================================
# UTILITIES
# =============================================================================


def get_least_used_cuda_device():
    """Select the CUDA device with the most free memory."""
    device_count = torch.cuda.device_count()
    if device_count == 1:
        return torch.device("cuda:0")
    
    max_free_memory = 0
    best_device = 0
    for i in range(device_count):
        free_memory, total_memory = torch.cuda.mem_get_info(i)
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_device = i
    
    return torch.device(f"cuda:{best_device}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = get_least_used_cuda_device()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Process experiment parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--architecture_pretty_name", type=str, required=True)
    parser.add_argument("--implant_type", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ridge_readout", "fine_tune_head", "fine_tune_all"],
        default="ridge_readout",
        help="Training mode: ridge_readout (linear Ridge on frozen features), "
             "fine_tune_head (train nn.Linear on frozen features), "
             "fine_tune_all (train full model end-to-end)"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    head_lr = config["head_lr"]
    backbone_lr = config["backbone_lr"]
    lr = config["lr"]
    num_runs = config["num_runs"]

    # Load train/test split
    split_path = os.path.join("..", "data", args.task, "train_test.json")
    with open(split_path, "r") as f:
        data_split = json.load(f)

    # Get train and test file information directly from JSON
    train_filenames = data_split[args.implant_type][args.model_type]["train_filenames"]
    train_labels = data_split[args.implant_type][args.model_type]["train_labels"]
    test_filenames = data_split[args.implant_type][args.model_type]["test_filenames"]
    test_labels = data_split[args.implant_type][args.model_type]["test_labels"]

    # Create label-to-index mapping from config and convert labels to indices
    label_list = config["tasks"][args.task]["labels"]
    label_to_idx = {label: idx for idx, label in enumerate(label_list)}
    train_labels = [label_to_idx[label] for label in train_labels]
    test_labels = [label_to_idx[label] for label in test_labels]

    zip_path = os.path.join("..", "data", args.task, "percepts.zip")

    # Sanity checks
    train_filenames_set = set(train_filenames)
    test_filenames_set = set(test_filenames)
    overlap = train_filenames_set.intersection(test_filenames_set)
    assert (
        len(overlap) == 0
    ), f"Data leakage detected: {len(overlap)} overlapping percept filenames between training and test sets."

    # -------------------------------------------------------------------------
    # MODE: ridge_readout
    # Frozen backbone features → sklearn RidgeClassifierCV
    # Single deterministic run (no random init), outputs to ridge_readout_predictions/
    # -------------------------------------------------------------------------
    if args.mode == "ridge_readout":
        run_ridge_readout(
            task_name=args.task,
            architecture=args.architecture,
            architecture_pretty_name=args.architecture_pretty_name,
            implant_type=args.implant_type,
            model_type=args.model_type,
            train_filenames=train_filenames,
            train_labels=train_labels,
            test_filenames=test_filenames,
            test_labels=test_labels,
            zip_path=zip_path,
            device=device,
            labels=label_list,
            batch_size=batch_size,
        )

    # -------------------------------------------------------------------------
    # MODE: fine_tune_head
    # Frozen backbone features → nn.Linear trained with AdamW
    # Multiple runs (seeded), outputs to head_model_predictions/ + loss curves
    # -------------------------------------------------------------------------
    elif args.mode == "fine_tune_head":
        run_fine_tune_head(
            task_name=args.task,
            architecture=args.architecture,
            architecture_pretty_name=args.architecture_pretty_name,
            implant_type=args.implant_type,
            model_type=args.model_type,
            train_filenames=train_filenames,
            train_labels=train_labels,
            test_filenames=test_filenames,
            test_labels=test_labels,
            zip_path=zip_path,
            device=device,
            labels=label_list,
            num_runs=num_runs,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
        )

    # -------------------------------------------------------------------------
    # MODE: fine_tune_all
    # Full model trained end-to-end (backbone_lr for backbone, head_lr for head)
    # Multiple runs (seeded), outputs to full_model_predictions/ + loss curves
    # -------------------------------------------------------------------------
    elif args.mode == "fine_tune_all":
        run_fine_tune_all(
            task_name=args.task,
            architecture=args.architecture,
            architecture_pretty_name=args.architecture_pretty_name,
            implant_type=args.implant_type,
            model_type=args.model_type,
            train_filenames=train_filenames,
            train_labels=train_labels,
            test_filenames=test_filenames,
            test_labels=test_labels,
            zip_path=zip_path,
            device=device,
            labels=label_list,
            num_runs=num_runs,
            num_epochs=num_epochs,
            batch_size=batch_size,
            head_lr=head_lr,
            backbone_lr=backbone_lr,
        )


