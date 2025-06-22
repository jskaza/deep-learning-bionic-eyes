import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'

from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from captum.attr import Saliency, LayerGradCam, IntegratedGradients
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
import zipfile
import random
import os
import json
from tqdm import tqdm
from utils import load_human_trials
from pathlib import Path
import matplotlib as mpl
import seaborn as sns


# Set font size
mpl.rcParams['font.size'] = 11

# Set the style to remove gridlines and keep only left and bottom axes
sns.set_style("ticks")
sns.despine(top=True, right=True)

# ----------------------------------------------------------------
#  GLOBALS
# ----------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ============  UTILITIES  ======================================

def denormalise(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=img.device)[:, None, None]
    std  = torch.tensor(IMAGENET_STD,  device=img.device)[:, None, None]
    return img * std + mean

def _read_image_from_zip(zf: zipfile.ZipFile, file_name: str) -> Image.Image:
    try:
        with zf.open(file_name) as fp:
            return Image.open(fp).convert('RGB')
    except KeyError:
        base_filename = file_name.split('/')[-1]
        all_files = zf.namelist()
        matching_files = [f for f in all_files if f.endswith(base_filename)]
        if matching_files:
            with zf.open(matching_files[0]) as fp:
                return Image.open(fp).convert('RGB')
        else:
            raise KeyError(f"Could not find {file_name} or any file ending with {base_filename}")

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def seed_worker(worker_id):
    """Worker init function for DataLoader to ensure reproducible results"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ============  DATASETS & TRANSFORMS  ==========================

def make_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


class ImageDataset(Dataset):
    def __init__(self, zip_path: str, fns: List[str], tfm: T.Compose):
        self.zip_path = zip_path
        self.fns = fns
        self.tfm = tfm
        self.zf = zipfile.ZipFile(zip_path, 'r')  # keep zip open

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, i):
        img = _read_image_from_zip(self.zf, self.fns[i])
        return self.tfm(img)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if hasattr(self, 'zf') and self.zf:
            self.zf.close()
            self.zf = None

    def __del__(self):
        self.close()


class LabeledDataset(ImageDataset):
    def __init__(self, zip_path, fns, labels, tfm):
        super().__init__(zip_path, fns, tfm); self.labels = labels
    def __getitem__(self, i):
        return super().__getitem__(i), self.labels[i], self.fns[i]

# ============  CUSTOM CNN MODEL  ===============================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # This will be our target layer for grad-cam
        self.target_layer = self.features[17]  # The last conv layer before ReLU
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Calculate output dim for the fully connected layers
        self.fc_input_dim = 128
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        features = self.gap(x).view(x.size(0), -1)
        return self.classifier(features)
    
    def get_features(self, x):
        x = self.features(x)
        return self.gap(x).view(x.size(0), -1)

# ============  ATTRIBUTION METHODS  ===========================

class SaliencyMapAttr:
    def __init__(self, model: nn.Module):
        self.saliency = Saliency(model)
        self.model = model.eval()

    def __call__(self, x: torch.Tensor, target_idx: int):
        # Make a grad-enabled copy of the input
        x_req = x.clone().detach().requires_grad_(True)

        # Compute saliency
        attribution = self.saliency.attribute(x_req, target=target_idx)
        
        # Take absolute value and sum across color channels
        attribution = attribution.abs().sum(dim=1, keepdim=True)
        
        # Normalize
        attribution = attribution / (attribution.max() + 1e-8)
        
        return attribution.squeeze().cpu().detach().numpy()

class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model.eval()
        self.gc = LayerGradCam(model, model.target_layer)
        
    def __call__(self, x: torch.Tensor, target_idx: int):
        # Compute attribution
        attribution = self.gc.attribute(x, target=target_idx, relu_attributions=True)
        
        # Apply ReLU to focus on features that have a positive impact on the target
        attribution = F.relu(attribution.sum(1, keepdim=True))
        
        # Normalize
        attribution = attribution / (attribution.max() + 1e-8)
        
        # Resize to match input size
        attribution = F.interpolate(attribution, x.shape[2:], mode='bilinear', align_corners=False)
        
        return attribution.squeeze().cpu().detach().numpy()

class IntegratedGradsAttr:
    def __init__(self, model: nn.Module):
        self.ig = IntegratedGradients(model)
        self.model = model.eval()
        
    def __call__(self, x: torch.Tensor, target_idx: int):
        # Create a baseline (black image)
        baseline = torch.zeros_like(x)
        
        # Compute integrated gradients
        attribution = self.ig.attribute(x, baseline, target=target_idx, n_steps=50)
        
        # Take absolute value and sum across color channels
        attribution = attribution.abs().sum(dim=1, keepdim=True)
        
        # Normalize
        attribution = attribution / (attribution.max() + 1e-8)
        
        return attribution.squeeze().cpu().detach().numpy()

def overlay(attribution, img):
    # Convert attribution to heatmap
    heat = plt.cm.magma(attribution)[...,:3]
    heat = (heat*255).astype(np.uint8)
    
    # Convert image to uint8
    img  = (img*255).astype(np.uint8)
    
    # Handle grayscale images
    if img.ndim==2:
        img = np.repeat(img[...,None],3,2)
        
    # Blend attribution and image
    return (0.7*heat + 0.3*img).astype(np.uint8)

def save_vis(img_t, attributions, labels, path):
    img_np = denormalise(img_t).clamp(0,1).permute(1,2,0).cpu().detach().numpy()
    
    # Create a figure with a row for each attribution method
    fig, axes = plt.subplots(len(attributions)+1, 1, figsize=(10, 4*len(attributions)))
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title(labels['orig'])
    axes[0].axis('off')
    
    # Attribution methods
    for i, (method_name, attribution) in enumerate(attributions.items(), 1):
        axes[i].imshow(overlay(attribution, img_np))
        axes[i].set_title(f"{method_name}: {labels[method_name.lower()]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close()

# ============  HUMAN CONSENSUS  ================================

def human_consensus(df: pd.DataFrame, min_subj=5):
    cnt = df.groupby(['percept_filename','implant_type','model_type']).subject.nunique()
    cnt = cnt[cnt>=min_subj].reset_index().rename(columns={'subject':'n'})
    rows=[]
    for _,r in cnt.iterrows():
        sub = df[(df.percept_filename==r.percept_filename)&
                 (df.implant_type==r.implant_type)&
                 (df.model_type  ==r.model_type)]
        rows.append({'percept_filename':r.percept_filename,'implant_type':r.implant_type,
                     'model_type':r.model_type,'human_consensus':sub.guess.mode()[0],
                     'ground_truth':sub.target.iloc[0],'num_subjects':r.n})
    return pd.DataFrame(rows)

# ============  MAIN EXPERIMENT  ================================

def leave_one_out(task, implant, mtype, device, cfg, sample_size=10):
    hdf = load_human_trials(task)
    hdf = hdf[(hdf.implant_type==implant)&(hdf.model_type==mtype)]
    if hdf.empty: return None
    cdf = human_consensus(hdf)
    if cdf.empty: return None
    elig = cdf[cdf.human_consensus != cdf.ground_truth]
    sel = elig.sample(n=min(sample_size,len(elig)), random_state=42)

    zip_path = Path(__file__).parent.parent / cfg['tasks'][task]['zip_path']
    tfm = make_transform()

    # Build label mappings once from the full dataset to ensure consistency across LOO iterations
    all_gt_labels = sorted(cdf.ground_truth.unique())
    all_hc_labels = sorted(cdf.human_consensus.unique())
    gt_map = {l:i for i,l in enumerate(all_gt_labels)}
    hc_map = {l:i for i,l in enumerate(all_hc_labels)}
    
    # Create reverse mappings
    gt_map_rev = {i:l for l,i in gt_map.items()}
    hc_map_rev = {i:l for l,i in hc_map.items()}

    # Storage for aggregated features
    rep_gt_all, rep_hc_all = [], []
    lab_gt_all, lab_hc_all = [], []
    corr_gt_all, corr_hc_all = [], []

    for k, row in sel.iterrows():
        train = cdf[cdf.percept_filename != row.percept_filename]

        # Check if training set contains all labels to avoid issues
        train_gt_labels = set(train.ground_truth.unique())
        train_hc_labels = set(train.human_consensus.unique())
        
        # Skip this iteration if training set is missing any labels
        if not train_gt_labels.issuperset(all_gt_labels):
            missing_gt = set(all_gt_labels) - train_gt_labels
            print(f"Warning: Skipping {row.percept_filename} - missing GT labels in training: {missing_gt}")
            continue
            
        if not train_hc_labels.issuperset(all_hc_labels):
            missing_hc = set(all_hc_labels) - train_hc_labels
            print(f"Warning: Skipping {row.percept_filename} - missing HC labels in training: {missing_hc}")
            continue

        # Create dataloaders for training
        def create_loader(col, m, sd):
            ds = LabeledDataset(zip_path, train.percept_filename.tolist(),
                                [m[l] for l in train[col]], tfm)
            g  = torch.Generator(); g.manual_seed(sd)
            return DataLoader(ds, 8, shuffle=True, num_workers=0, pin_memory=True,
                              generator=g)

        # Create our custom CNN models
        gt_model = SimpleCNN(num_classes=len(gt_map))
        hc_model = SimpleCNN(num_classes=len(hc_map))
        
        # Move models to device
        gt_model.to(device)
        hc_model.to(device)
        
        # Train ground truth model
        train_loader_gt = create_loader('ground_truth', gt_map, 100+k)
        gt_optimizer = optim.AdamW(gt_model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        for epoch in tqdm(range(1, 31), desc=f'Training GT model {row.percept_filename}'):
            gt_model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for images, labels, _ in train_loader_gt:
                images, labels = images.to(device), labels.to(device)
                
                gt_optimizer.zero_grad()
                outputs = gt_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                gt_optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            print(f'  GT Epoch {epoch}/30 | Loss: {total_loss/len(train_loader_gt):.4f} | Acc: {100*correct/total:.1f}%')
        
        # Close GT dataset zipfile handle
        train_loader_gt.dataset.close()
        
        # Train human consensus model
        train_loader_hc = create_loader('human_consensus', hc_map, 200+k)
        hc_optimizer = optim.AdamW(hc_model.parameters())
        
        for epoch in tqdm(range(1, 31), desc=f'Training HC model {row.percept_filename}'):
            hc_model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for images, labels, _ in train_loader_hc:
                images, labels = images.to(device), labels.to(device)
                
                hc_optimizer.zero_grad()
                outputs = hc_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                hc_optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            print(f'  HC Epoch {epoch}/30 | Loss: {total_loss/len(train_loader_hc):.4f} | Acc: {100*correct/total:.1f}%')
        
        # Close HC dataset zipfile handle
        train_loader_hc.dataset.close()
        
        single_img_ds = ImageDataset(zip_path, [row.percept_filename], tfm)
        heldout_img = single_img_ds[0].unsqueeze(0).to(device)

        # Process with ground truth model
        gt_model.eval()
        with torch.no_grad():
            features = gt_model.get_features(heldout_img)
            rep_gt = features.cpu().numpy()
            outputs = gt_model(heldout_img)
            pred_gt = outputs.argmax(1).item()
            correct_gt = int(pred_gt == gt_map[row.ground_truth])
            label_gt = row.ground_truth

        # Process with human consensus model
        hc_model.eval()
        with torch.no_grad():
            features = hc_model.get_features(heldout_img)
            rep_hc = features.cpu().numpy()
            outputs = hc_model(heldout_img)
            pred_hc = outputs.argmax(1).item()
            correct_hc = int(pred_hc == hc_map[row.human_consensus])
            label_hc = row.human_consensus
         
        # Generate multiple attribution visualizations
        try:
            # Initialize attribution methods for ground truth model
            print(f"\nGenerating attributions for {row.percept_filename}")
            print("Initializing attribution methods...")
            
            gt_saliency = SaliencyMapAttr(gt_model)
            gt_gradcam = GradCAM(gt_model)
            gt_ig = IntegratedGradsAttr(gt_model)
            
            print("Generating ground truth model attributions...")
            # Check if input tensor has valid values
            print(f"Input tensor stats - min: {heldout_img.min():.3f}, max: {heldout_img.max():.3f}, mean: {heldout_img.mean():.3f}")
            
            # Generate ground truth model attributions
            gt_attributions = {
                "Saliency": gt_saliency(heldout_img, pred_gt),
                "GradCAM": gt_gradcam(heldout_img, pred_gt),
                "IntegratedGrads": gt_ig(heldout_img, pred_gt)
            }
            
            print("Ground truth attributions generated successfully")
            
            # Initialize attribution methods for human consensus model
            print("Initializing human consensus attribution methods...")
            hc_saliency = SaliencyMapAttr(hc_model)
            hc_gradcam = GradCAM(hc_model)
            hc_ig = IntegratedGradsAttr(hc_model)
            
            print("Generating human consensus model attributions...")
            # Generate human consensus model attributions
            hc_attributions = {
                "Saliency": hc_saliency(heldout_img, pred_hc),
                "GradCAM": hc_gradcam(heldout_img, pred_hc),
                "IntegratedGrads": hc_ig(heldout_img, pred_hc)
            }
            
            print("Human consensus attributions generated successfully")
            
            # Create comparison visualizations
            print("Creating comparison visualizations...")
            print(gt_map_rev[pred_gt], label_gt, hc_map_rev[pred_hc], label_hc)
            if gt_map_rev[pred_gt] == label_gt and hc_map_rev[pred_hc] == label_hc:
                for method_name in gt_attributions.keys():
                    comparison = {
                        f"Ground Truth {method_name}": gt_attributions[method_name],
                        f"Human Consensus {method_name}": hc_attributions[method_name]
                    }
                    save_vis(
                        heldout_img.squeeze(0),
                        comparison,
                        {
                            'orig': f"Original Percept: Ground Truth→{label_gt}, Human Consensus→{label_hc}",
                            f'ground truth {method_name.lower()}': f"Ground Truth {gt_map_rev[pred_gt]}",
                            f'human consensus {method_name.lower()}': f"Human Consensus {hc_map_rev[pred_hc]}"
                        },
                        f"plots/attributions/{task}_{implant}_{mtype}_{row.percept_filename.replace('.tif','')}_compare_{method_name.lower()}.pdf"
                    )
                print("Visualizations saved successfully")
            
        except Exception as e:
            print(f"Attribution error details: {str(e)}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            
        # Close single image dataset zipfile handle
        single_img_ds.close()
            
        # Store results
        rep_gt_all.append(rep_gt)
        rep_hc_all.append(rep_hc)
        lab_gt_all.append(label_gt)
        lab_hc_all.append(label_hc)
        corr_gt_all.append(correct_gt)
        corr_hc_all.append(correct_hc)

        # Clean up GPU memory after each iteration
        del gt_model, hc_model
        torch.cuda.empty_cache()

    return {
        'task': task, 'implant': implant, 'model_type': mtype,
        'rep_gt': np.vstack(rep_gt_all),
        'rep_hc': np.vstack(rep_hc_all),
        'label_gt': lab_gt_all, 'label_hc': lab_hc_all,
        'correct_gt': corr_gt_all, 'correct_hc': corr_hc_all
    }


# ============  ENTRY POINT  ====================================

def main():
    set_seed()
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f'Using device: {device}')

    cfg = json.load((Path(__file__).resolve().parent / '..' / 'config.json').open())
    all_reps = []

    for task, tcfg in [("doorway",cfg['tasks']['doorway']),("emotion",cfg['tasks']['emotion']),("shape",cfg['tasks']['shape'])]:
        if 'subject_data_path' not in tcfg: continue
        split = json.load(open(Path(__file__).parent.parent / tcfg['data_split_path']))

        for implant, md in split.items():
            if implant != '12_20': continue
            for mtype in md.keys():
                if mtype != 'streaky': continue
                result = leave_one_out(task, implant, mtype, device, cfg)
                if result:
                    all_reps.append(result)
                    torch.cuda.empty_cache()

if __name__=='__main__':
    main()
