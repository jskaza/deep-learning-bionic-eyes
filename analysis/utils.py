import os
import json
import gzip
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm import tqdm

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Build label mappings from config: {task: {label_str: index}}
label_mappings = {
    task: {label: idx for idx, label in enumerate(task_config["labels"])}
    for task, task_config in config["tasks"].items()
}

def load_model_predictions(task: str, implant_type: str, model_type: str, head_only: bool = False) -> pd.DataFrame:
    """
    Loads all model predictions and adds implant_type, model_type, architecture, run to each entry.
    Merges with metadata from CSV file specified in config.
    Returns a DataFrame.
    """
    all_preds = []
    patterns = list(glob(os.path.join(os.path.dirname(__file__), task, "head_model_predictions", implant_type, model_type, "*", "*.json.gz")))
    if not head_only:
        patterns.extend(list(glob(os.path.join(os.path.dirname(__file__), task, "full_model_predictions", implant_type, model_type, "*", "*.json.gz"))))
    
    for filepath in patterns:
        with gzip.open(filepath, 'rt') as f:
            preds = pd.read_json(f)
            preds["task"] = task
            preds["method"] = filepath.split("/")[-5].replace("_predictions", "")
            preds['implant_type'] = filepath.split("/")[-4]
            preds['model_type'] = filepath.split("/")[-3]
            preds['architecture'] = filepath.split("/")[-2].split(" (")[0]
            preds['run'] = int(filepath.split("/")[-1].split(".")[0])
            preds['numeric_label'] = preds['target'].map(label_mappings[task])
            preds['numeric_pred'] = preds['probabilities'].apply(lambda x: x.index(max(x)))
            all_preds.append(preds)
    
    # Concatenate all predictions
    df = pd.concat(all_preds) if all_preds else pd.DataFrame()

    return df

def load_human_trials(task: str) -> pd.DataFrame:
    """
    Loads human trial data if available in the config.
    Also adds percept_filename using human_to_model filename mapping if available.
    Returns an empty DataFrame if subject_data_path is not in the config.
    """
    config_path = Path(__file__).parent.parent / 'config.json'
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check if subject_data_path exists for this task
        if task in config['tasks'] and 'subject_data_path' in config['tasks'][task]:
            human_path = Path(task) / "human_trials" / "human_trials.csv"
            if human_path.exists():
                df = pd.read_csv(human_path)
                df["architecture"] = "human"
                df["architecture_family"] = "human"
                df["run"] = df["subject"]
                # Apply filename mapping if available
                if 'filename_mapping' in config['tasks'][task]:
                    mapping_path = Path(__file__).parent.parent / config['tasks'][task]['filename_mapping']
                    try:
                        with open(mapping_path, 'r') as f:
                            mapping_data = json.load(f)
                        
                        # Get human_to_model mapping
                        filename_map = mapping_data.get('human_to_model', {})
                        if filename_map and 'percept_filename' in df.columns:
                            # Map human filenames to model filenames
                            df['percept_filename'] = df['percept_filename'].map(
                                lambda f: filename_map.get(f, f)
                            )
                    except Exception as e:
                        print(f"Error applying filename mapping: {e}")
                
                return df
    except Exception as e:
        print(f"Error loading human trials: {e}")
    
    # Return empty DataFrame if subject_data_path not in config or file doesn't exist
    return pd.DataFrame()