"""
Generate a mapping between filenames in pre_changes metadata and current metadata.
This mapping allows matching between human trial data and model predictions.
"""
import os
import json
import pandas as pd

def create_filename_mapping():
    """
    Creates a mapping between filenames in pre_changes metadata and current metadata
    by matching on stimulus parameters rather than filenames directly.
    
    Returns:
        dict: A mapping with two keys:
            - 'model_to_human': Maps model filenames to human filenames
            - 'human_to_model': Maps human filenames to model filenames
    """
    # Load both metadata files
    pre_changes_path = '/home/jonathan/Argus/data/data_pre_changes/doorway_metadata.csv'
    current_path = '/home/jonathan/Argus/data/doorway/metadata.csv'
    
    print(f"Loading pre-changes metadata from {pre_changes_path}")
    pre_changes_metadata = pd.read_csv(pre_changes_path)
    
    print(f"Loading current metadata from {current_path}")
    current_metadata = pd.read_csv(current_path)
    
    # Check which columns we have in common to match on
    pre_changes_cols = set(pre_changes_metadata.columns)
    current_cols = set(current_metadata.columns)
    common_cols = pre_changes_cols.intersection(current_cols)
    
    # Remove filename-related columns and any source columns
    exclude_cols = {'percept_filename', 'data_filename', 'source'}
    match_cols = [col for col in common_cols if col not in exclude_cols]
    
    print(f"Matching on columns: {match_cols}")
    
    # Create bidirectional mappings
    model_to_human = {}
    human_to_model = {}
    
    # Merge the dataframes on matching columns
    print("Merging dataframes on matching columns")
    merged = pd.merge(
        pre_changes_metadata[match_cols + ['percept_filename']],
        current_metadata[match_cols + ['percept_filename']],
        on=match_cols,
        suffixes=('_human', '_model')
    )
    
    # Create mappings from the merged dataframe
    for _, row in merged.iterrows():
        human_file = row['percept_filename_human']
        model_file = row['percept_filename_model']
        model_to_human[model_file] = human_file
        human_to_model[human_file] = model_file
    
    print(f"Created mapping with {len(model_to_human)} entries")
    
    return {
        'model_to_human': model_to_human,
        'human_to_model': human_to_model
    }

def main():
    mapping = create_filename_mapping()
    
    # Save the mapping to a JSON file
    output_path = '/home/jonathan/Argus/data/doorway/mappings/filename_mapping.json'
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