import os
import json
import gzip
import pandas as pd
from glob import glob
from pathlib import Path

def load_model_predictions(task: str) -> pd.DataFrame:
    """
    Loads all model predictions and adds implant_type, model_type, architecture, run to each entry.
    Merges with metadata from CSV file specified in config.
    Returns a DataFrame.
    """
    all_preds = []
    pattern = os.path.join(task, 'model_predictions', '*', '*', '*', '*.json.gz')
    file_paths = glob(pattern)
    
    for filepath in file_paths:
        parts = filepath.split(os.sep)
        # .../model_predictions/{implant_type}/{model_type}/{architecture}/{run}.json.gz
        implant_type = parts[-4]
        model_type = parts[-3]
        architecture = parts[-2]
        run = os.path.splitext(os.path.basename(filepath))[0]
        with gzip.open(filepath, 'rt') as f:
            preds = json.load(f)
            for entry in preds:
                entry['implant_type'] = implant_type
                entry['model_type'] = model_type
                entry['architecture'] = architecture
                entry['run'] = int(run.split(".")[0])
                all_preds.append(entry)
    
    # Create a basic DataFrame from all predictions
    df = pd.DataFrame(all_preds)
    df['task'] = task
    # Ensure percept_filename exists (same as filename)
    if 'filename' in df.columns and df.shape[0] > 0:
        df['percept_filename'] = df['filename']
    
    # Load and merge metadata from config
    config_path = Path(__file__).parent.parent / 'config.json'
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if task in config['tasks'] and 'metadata' in config['tasks'][task]:
            metadata_path = config['tasks'][task]['metadata']
            if os.path.exists(metadata_path):
                metadata_df = pd.read_csv(metadata_path)
                # Merge based on percept_filename and potentially model_type, implant_type
                merge_cols = ['percept_filename']
                
                # Add model_type and implant_type to merge columns if they exist in metadata
                if 'model_type' in metadata_df.columns:
                    merge_cols.append('model_type')
                if 'implant_type' in metadata_df.columns:
                    merge_cols.append('implant_type')
                    
                if 'percept_filename' in df.columns and 'percept_filename' in metadata_df.columns:
                    df = pd.merge(df, metadata_df, on=merge_cols, how='left')
    except Exception as e:
        print(f"Error loading or merging metadata: {e}")
        # Create a mapping from architecture to model family
    architecture_to_family = {}
    for family, architectures in config['architecture_groupings']['By Model Family'].items():
        for arch in architectures:
            architecture_to_family[arch] = family
    df["architecture_family"] = df["architecture"].map(architecture_to_family)

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