"""
Data Processing

This script processes human trial data and model predictions to generate 
consolidated datasets for analysis. It computes accuracy metrics, including 
normalized above-chance accuracy, for both human subjects and computational 
models across various tasks and implant types.

Output Datasets:
    - human_accs.csv: Subject-level accuracies for human trials
    - human_trials.csv: Raw trial-level human data
    - model_accs.csv: Run-level accuracies for all model configurations
    - model_accs_human_subset.csv: Model accuracies matching human experiment configs
    - model_predictions_human_subset.pkl: Detailed predictions with logits/probabilities
"""

import json
import os
import numpy as np
import pandas as pd
from utils import load_human_trials, load_model_predictions


def load_config():
    """
    Load the project configuration file.
    
    Returns:
        dict: Configuration containing tasks, implant types, and model types
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def compute_chance_levels(config):
    """
    Compute chance-level accuracy for each task based on number of classes.
    
    Args:
        config (dict): Configuration dictionary containing task definitions
        
    Returns:
        dict: Mapping of task names to chance-level accuracy (1/n_classes)
    """
    return {
        task: 1.0 / len(config['tasks'][task]['labels'])
        for task in config['tasks']
    }


def add_normalized_accuracy(df, chance_levels):
    """
    Add normalized above-chance accuracy to a dataframe.
    
    Normalized accuracy is computed as: (acc - chance) / (1 - chance)
    This metric ranges from 0 (chance performance) to 1 (perfect performance).
    
    Args:
        df (pd.DataFrame): Dataframe with 'task' and 'accuracy' columns
        chance_levels (dict): Mapping of task names to chance-level accuracy
        
    Returns:
        pd.DataFrame: Input dataframe with added 'chance' and 'accuracy_above_chance' columns
    """
    df["chance"] = df["task"].map(chance_levels)
    df["accuracy_above_chance"] = (
        (df["accuracy"] - df["chance"]) / (1 - df["chance"])
    )
    return df


# Initialize directories and configuration
config = load_config()
chance_levels = compute_chance_levels(config)
datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")
os.makedirs(datasets_dir, exist_ok=True)

################################################################################
# Process Human Trial Data
################################################################################
print("Processing human trial data...")

# Load trials only for tasks that have human subject data
tasks_with_subjects = [
    task for task in config['tasks'] 
    if "subjects" in config['tasks'][task]
]
human_trials = pd.concat([
    load_human_trials(task) for task in tasks_with_subjects
])

# Aggregate trials to compute per-subject accuracy
# Each row represents one subject's performance on one task/implant/model combination
human_accs = human_trials.groupby(
    ["task", "implant_type", "model_type", "run"]
).agg(
    accuracy=("result", "mean"),
).reset_index()

# Add normalized above-chance accuracy metric
human_accs = add_normalized_accuracy(human_accs, chance_levels)

# Extract unique task/implant/model combinations for later model comparison
human_configs = human_accs[["task", "implant_type", "model_type"]].drop_duplicates()

# Save processed datasets
human_accs.to_csv(os.path.join(datasets_dir, "human_accs.csv"), index=False)
human_trials.to_csv(os.path.join(datasets_dir, "human_trials.csv"), index=False)

print(f"  ✓ Saved {len(human_accs)} subject-level accuracies to datasets/human_accs.csv")
print(f"  ✓ Saved {len(human_trials)} trials to datasets/human_trials.csv")

################################################################################
# Process Model Predictions (All Configurations)
################################################################################
print("\nProcessing model predictions (all configurations)...")

# Load predictions for all possible task/implant/model combinations
# Using head_only=True loads only the task-specific classification head predictions
model_predictions_all = []
for task in config["tasks"]:
    for implant_type in config["implant_types"]:
        for model_type in config["model_types"]:
            model_predictions_all.append(
                load_model_predictions(task, implant_type, model_type, head_only=True)
            )

model_predictions_all = pd.concat(model_predictions_all).reset_index(drop=True)

# Aggregate predictions to compute per-run accuracy
# Each row represents one model run's performance on one task/implant/model/architecture combination
model_accs_all = model_predictions_all.groupby(
    ["task", "implant_type", "model_type", "architecture", "method", "run"]
).agg(
    accuracy=("result", "mean"),
).reset_index()

# Add normalized above-chance accuracy metric
model_accs_all = add_normalized_accuracy(model_accs_all, chance_levels)

# Save comprehensive model accuracy dataset
model_accs_all.to_csv(os.path.join(datasets_dir, "model_accs.csv"), index=False)
print(f"  ✓ Saved {len(model_accs_all)} model run accuracies to datasets/model_accs.csv")

################################################################################
# Process Model Predictions (Human Experiment Subset)
################################################################################
print("\nProcessing model predictions (human experiment subset)...")

# Load predictions only for task/implant/model combinations that were tested with humans
# This enables direct comparison between human and model performance
# Note: head_only=False to get full predictions with logits and probabilities
model_predictions_human_subset = pd.concat([
    load_model_predictions(row["task"], row["implant_type"], row["model_type"])
    for _, row in human_configs.iterrows()
])

# Aggregate predictions to compute per-run accuracy
model_accs_human_subset = model_predictions_human_subset.groupby(
    ["task", "implant_type", "model_type", "architecture", "method", "run"]
).agg(
    accuracy=("result", "mean"),
).reset_index()

# Add normalized above-chance accuracy metric
model_accs_human_subset = add_normalized_accuracy(
    model_accs_human_subset, chance_levels
)

# Save accuracy dataset for human-model comparison
model_accs_human_subset.to_csv(
    os.path.join(datasets_dir, "model_accs_human_subset.csv"), index=False
)
print(f"  ✓ Saved {len(model_accs_human_subset)} model run accuracies to datasets/model_accs_human_subset.csv")

# Prepare detailed predictions for saving
# Convert logits and probabilities to numpy arrays for efficient storage
model_predictions_human_subset["logits"] = model_predictions_human_subset["logits"].apply(np.array)
model_predictions_human_subset["probabilities"] = model_predictions_human_subset["probabilities"].apply(np.array)
model_predictions_human_subset["percept_filename"] = model_predictions_human_subset["filename"]

# Save detailed predictions as pickle for preserving numpy arrays and full data structure
model_predictions_human_subset.to_pickle(
    os.path.join(datasets_dir, "model_predictions_human_subset.pkl")
)
print(f"  ✓ Saved {len(model_predictions_human_subset)} detailed predictions to datasets/model_predictions_human_subset.pkl")

print("\n✓ Data processing complete!")
