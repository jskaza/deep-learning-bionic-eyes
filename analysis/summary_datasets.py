from utils import load_human_trials, load_model_predictions
import os
import json
import pandas as pd
import pickle
# Load and merge metadata from config
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
os.makedirs("summary_datasets", exist_ok=True)

model_predictions = pd.concat([load_model_predictions(task) for task in config['tasks']])
human_trials = pd.concat([load_human_trials(task) for task in config['tasks'] if "subjects" in config['tasks'][task]])
human_trials.groupby(["task", "implant_type", "model_type", "subject"]).agg(
    accuracy=("result", "mean"),
).reset_index().to_csv(os.path.join("summary_datasets", "accuracy_by_task_implant_model_subject.csv"), index=False)
model_predictions.groupby(["task", "implant_type", "model_type", "run"]).agg(
    accuracy=("result", "mean"),
).reset_index().to_csv(os.path.join("summary_datasets", "accuracy_by_task_implant_model_run.csv"), index=False)
# Run-level accuracy by task, implant type, model type, architecture, and run
run_level_accuracy_model = model_predictions.groupby(["task", "implant_type", "model_type", "architecture", "run"]).agg(
    architecture_family=("architecture_family", "first"),
    accuracy=("result", "mean"),
    trial_count=("result", "count")
).reset_index()
run_level_accuracy_human = human_trials.groupby(["task", "implant_type", "model_type", "architecture", "run"]).agg(
    architecture_family=("architecture_family", "first"),
    accuracy=("result", "mean"),
    trial_count=("result", "count")
).reset_index()
accuracy_by_task_implant_model_architecture_run = pd.concat([run_level_accuracy_model, run_level_accuracy_human])
accuracy_by_task_implant_model_architecture_run.to_csv(os.path.join("summary_datasets", "accuracy_by_task_implant_model_architecture_run.csv"), index=False)
print("Run-level accuracy by task, implant type, model type, architecture, and run")
print(accuracy_by_task_implant_model_architecture_run.head().to_markdown())


# Median accuracy by task and architecture
median_accuracy_model = run_level_accuracy_model.groupby(["task", "architecture"]).agg(
    accuracy=("accuracy", "median"),
    instances=("run", "nunique"),
).reset_index()
median_accuracy_human = run_level_accuracy_human.groupby(["task", "architecture"]).agg(
    accuracy=("accuracy", "median"),
    instances=("run", "nunique"),
).reset_index()
accuracy_by_task_architecture = pd.concat([median_accuracy_model, median_accuracy_human])
accuracy_by_task_architecture.to_csv(os.path.join("summary_datasets", "accuracy_by_task_architecture.csv"), index=False)
print("Median accuracy by task and architecture")
print(accuracy_by_task_architecture.head().to_markdown())

# Accuracy by stimulus
MIN_SAMPLES = 5
# Create dictionaries mapping architectures/humans to stimuli to result arrays
arch_to_stim2results = {}
# Process model predictions
for arch, grp in model_predictions.groupby("architecture_family"):
    grp["stim_id"] = grp["task"] + "_" + grp["percept_filename"]
    stim2results = {
        stim: g["result"].to_numpy()
        for stim, g in grp.groupby("stim_id")
    }
    # Filter stimuli with too few samples
    stim2results = {
        stim: results for stim, results in stim2results.items() 
        if len(results) >= MIN_SAMPLES
    }
    arch_to_stim2results[arch] = stim2results
# Process human trials (add as a special "architecture")
human_trials["stim_id"] = human_trials["task"] + "_" + human_trials["percept_filename"]
stim2human = {
    stim: grp["result"].to_numpy()
    for stim, grp in human_trials.groupby("stim_id")
}
# Filter stimuli with too few samples
stim2human = {
    stim: results for stim, results in stim2human.items() 
    if len(results) >= MIN_SAMPLES
}
arch_to_stim2results["human"] = stim2human
# Save the dictionary to a file
with open(os.path.join("summary_datasets", "arch_to_stim2results.pkl"), "wb") as f:
    pickle.dump(arch_to_stim2results, f)












