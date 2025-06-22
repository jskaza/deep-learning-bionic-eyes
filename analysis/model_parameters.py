import json
import timm
import pandas as pd
import torch
import torch.nn as nn

def get_feature_dim_and_params(arch):
    """Get feature dimension and calculate trainable parameters for each architecture."""
    # Create model without classifier to get feature dimension
    model = timm.create_model(arch, pretrained=True, num_classes=0)
    # Get a sample input
    dummy_input = torch.randn(1, 3, 224, 224)
    # Get feature dimension
    with torch.no_grad():
        features = model(dummy_input)
        if len(features.shape) > 2:  # Handle spatial dimensions if present
            features = features.mean([2, 3]) if len(features.shape) == 4 else features.reshape(features.size(0), -1)
    feature_dim = features.shape[1]
    
    # Calculate parameters for different numbers of classes
    params_by_classes = {}
    for num_classes in [2, 3]:  # Most tasks have 2 or 3 classes
        linear_layer = nn.Linear(feature_dim, num_classes)
        trainable_params = sum(p.numel() for p in linear_layer.parameters())
        params_by_classes[num_classes] = trainable_params
    
    return feature_dim, params_by_classes

def main():
    # Load config file
    with open("../config.json", "r") as f:
        config = json.load(f)
    
    # Get list of architectures
    architectures = config["architectures"]
    
    # Store data for DataFrame
    data = []
    
    for arch in architectures:
        feature_dim, params_by_classes = get_feature_dim_and_params(arch)
        
        # Store data for DataFrame
        data.append({
            'Architecture': arch,
            'Feature Dimension': feature_dim,
            'Trainable Params (2 classes)': f"{params_by_classes[2]:,}",
            'Trainable Params (3 classes)': f"{params_by_classes[3]:,}"
        })
    
    # Create DataFrame and display
    df = pd.DataFrame(data)
    df.sort_values(by='Trainable Params (2 classes)', ascending=False, inplace=True)
    print("\nTrainable Parameters per Architecture:")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    print(df.to_string(index=False))
    


if __name__ == "__main__":
    main() 