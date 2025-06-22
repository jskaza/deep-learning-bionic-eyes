import pandas as pd
import json
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process human trial data for specific tasks')
parser.add_argument('--tasks', type=str, nargs='+', default=['shape', 'emotion', 'doorway'], help='Tasks to process (e.g., shape doorway emotion)')
parser.add_argument('--config', type=str, default=str(Path(__file__).parent.parent / 'config.json'), help='Path to the config file')
args = parser.parse_args()

# Load configuration from JSON file
with open(args.config, "r") as f:
    config = json.load(f)

# Dictionary to store summary statistics
summary_stats = {}

# Process each task sequentially
for task in args.tasks:
    print(f"\nProcessing task: {task}")
    
    # Check if the specified task exists in the config
    if task not in config["tasks"]:
        print(f"Error: Task '{task}' not found in config. Available tasks: {', '.join(config['tasks'].keys())}")
        continue  # Skip to next task instead of exiting

    # Get the task specifications
    specs = config["tasks"][task]

    # initialize lists to store dataframes
    subject_data_dfs = []
    percept_dfs = []

    # Process subject data if available
    if "subject_data_path" in specs:
        IDKey = specs["IDKey"]
        files_list = []
        guess_list = []
        target_list = []
        implant_types_list = []
        model_types_list = []
        subjects_list = []
        
        # Iterate through all combinations of subjects, implant types, and model types
        for s in config["tasks"][task]["subjects"]:
            for i in config["implant_types"]:
                for m in config["model_types"]:
                    subject_data_path = Path(__file__).parent.parent / specs['subject_data_path'].format(subject=s, implant=i, model=m)
                    if not subject_data_path.exists():
                        # print(f"Task {task} Subject {s} implant {i} model {m} does not exist")
                        continue
                    
                    # Load and process subject data
                    with open(subject_data_path, "r") as f:
                        data = json.load(f)
                        trial_order = data.get("Data").get("TrialOrder")
                        target = np.array(data.get("Data").get("InputData").get("Feedback"))[trial_order]
                        files = np.array(data.get("Data").get("InputData").get("ImagePaths"))[trial_order]
                        files = [f.split("\\")[-1] for f in files]
                        guess = np.array(data.get("Data").get(IDKey))
                        # Handle cases where guess and files arrays have matching lengths
                        if len(guess) == len(files):
                            accuracy = np.array(guess == target)
                            implant_types_list += [i] * len(files)
                            model_types_list += [m] * len(files)
                            files_list += files
                            guess_list += guess.tolist()
                            target_list += target.tolist()
                            subjects_list += [s] * len(files)
                        # Handle cases where there are fewer guesses than files
                        elif len(guess) < len(files):
                            accuracy = np.array(guess == target[:len(guess)])
                            implant_types_list += [i] * len(guess)
                            model_types_list += [m] * len(guess)
                            files_list += files[:len(guess)]
                            guess_list += guess.tolist()
                            target_list += target[:len(guess)].tolist()
                            subjects_list += [s] * len(guess)
                            print(f"Only using {len(guess)} trials for {s} {i} {m}")
                        # Skip cases where there are more guesses than files
                        else:
                            print(f"More trials than guesses for {s} {i} {m}, skipping")
                            continue
                            

        # Create a dataframe from collected data
        data = {
            "percept_filename": files_list,
            "guess": guess_list,
            "target": target_list,
            "implant_type": implant_types_list,
            "model_type": model_types_list,
            "subject": subjects_list
        }
        df = pd.DataFrame(data)
        df["result"] = df["guess"] == df["target"]
        df["task"] = task
        subject_data_dfs.append(df)
        

    # Combine dataframes if there are any
    if subject_data_dfs:
        subject_data_df = pd.concat(subject_data_dfs)
        
        # Calculate summary statistics
        n_subjects = len(subject_data_df['subject'].unique())
        n_trials = int(subject_data_df.groupby(['subject','implant_type','model_type'])['percept_filename'].count().mode()[0])
        summary_stats[task] = {
            'n_subjects': n_subjects,
            'n_trials': n_trials
        }
        
        # Determine output file path
        output_path = f"{task}/human_trials/human_trials.csv"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the dataframe to CSV
        subject_data_df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    
        # Get ordered implant types from config
        ordered_implant_types = config["implant_types"]
        
        # Create combined category
        subject_data_df['combined_type'] = subject_data_df['implant_type'] + '_' + subject_data_df['model_type']
        # Calculate accuracy for each combination
        combined_accuracy = subject_data_df.groupby('combined_type')['result'].mean()
        # Sort by implant type order first, then model type
        combined_accuracy.index = pd.Categorical(
            combined_accuracy.index,
            categories=[f"{i}_{m}" for i in ordered_implant_types for m in config["model_types"]],
            ordered=True
        )
        combined_accuracy = combined_accuracy.sort_index()
        # Plot
        plt.figure(figsize=(15, 10))
        # Create pivot table with subject vs combined type
        pivot_table = subject_data_df.pivot_table(
            values='result',
            index='subject',
            columns='combined_type',
            aggfunc='mean'
        )
        # Reorder columns to match implant type order first, then model type
        column_order = [f"{i}_{m}" for i in ordered_implant_types for m in config["model_types"]]
        pivot_table = pivot_table.reindex(columns=column_order)
        # Create heatmap
        sns.heatmap(pivot_table, annot=True, cmap='PiYG', fmt='.2f', vmin=0, vmax=1)
        plt.title(f'Accuracy by Subject and Implant-Model Type - {task}')
        plt.xlabel('Implant Type - Model Type')
        plt.ylabel('Subject')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{task}/human_trials/accuracy_by_subject_combined_type.png")
        plt.close()
        
        print(f"Plots saved to {task}/human_trials directory")
    else:
        print(f"No subject data found for task '{task}'")

# Save summary statistics
summary_path = "summary_datasets/human_trials.json"
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, "w") as f:
    json.dump(summary_stats, f, indent=4)
print(f"\nSummary statistics saved to {summary_path}")
