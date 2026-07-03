"""
Generate a mapping between filenames in pre_changes metadata and current metadata for emotion data.
This mapping allows matching between human trial data and model predictions.
"""
import os
import json
import pandas as pd

def create_filename_mapping():
    """
    Creates a mapping between filenames in pre_changes metadata and current metadata
    by matching on implant_type, model_type, and data_filename.
    
    Returns:
        dict: A mapping with two keys:
            - 'model_to_human': Maps model filenames to human filenames
            - 'human_to_model': Maps human filenames to model filenames
    """
    # Load both metadata files
    pre_changes_path = '/home/jonathan/Argus/data/data_pre_changes/emotion_metadata.csv'
    current_path = '/home/jonathan/Argus/data/emotion/metadata.csv'
    
    print(f"Loading pre-changes metadata from {pre_changes_path}")
    pre_changes_metadata = pd.read_csv(pre_changes_path)
    
    print(f"Loading current metadata from {current_path}")
    current_metadata = pd.read_csv(current_path)
    
    # Create bidirectional mappings
    model_to_human = {}
    human_to_model = {}
    
    # Create a composite key with implant_type, model_type, and data_filename
    print("Creating mapping based on implant_type, model_type, and data_filename columns")
    
    # Add composite key to both dataframes
    pre_changes_metadata['composite_key'] = pre_changes_metadata.apply(
        lambda row: f"{row.get('implant_type', 'unknown')}_{row.get('model_type', 'unknown')}_{row['data_filename']}", 
        axis=1
    )
    current_metadata['composite_key'] = current_metadata.apply(
        lambda row: f"{row.get('implant_type', 'unknown')}_{row.get('model_type', 'unknown')}_{row['data_filename']}", 
        axis=1
    )
    
    # Group both dataframes by the composite key
    pre_changes_grouped = pre_changes_metadata.groupby('composite_key')['percept_filename'].apply(list).to_dict()
    current_grouped = current_metadata.groupby('composite_key')['percept_filename'].apply(list).to_dict()
    
    # Find common composite keys
    common_keys = set(pre_changes_grouped.keys()).intersection(set(current_grouped.keys()))
    print(f"Found {len(common_keys)} common entries matching on implant_type, model_type, and data_filename")
    
    # For each common key, map all pairs of percept filenames
    for composite_key in common_keys:
        human_files = pre_changes_grouped[composite_key]
        model_files = current_grouped[composite_key]
        
        for model_file in model_files:
            for human_file in human_files:
                model_to_human[model_file] = human_file
                human_to_model[human_file] = model_file
    
    print(f"Created mapping with {len(model_to_human)} entries")
    
    return {
        'model_to_human': model_to_human,
        'human_to_model': human_to_model
    }

def main():
    mapping = create_filename_mapping()
    
    # Create the directory if it doesn't exist
    os.makedirs('/home/jonathan/Argus/data/emotion/mappings', exist_ok=True)
    
    # Save the mapping to a JSON file
    output_path = '/home/jonathan/Argus/data/emotion/mappings/filename_mapping.json'
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Saved mapping to {output_path}")
    
    # Print a few examples from the mapping
    model_to_human = mapping['model_to_human']
    if model_to_human:
        print("\nExample mappings (model → human):")
        count = 0
        for model_file, human_file in model_to_human.items():
            print(f"  {model_file} → {human_file}")
            count += 1
            if count >= 5:
                break

if __name__ == '__main__':
    main() 