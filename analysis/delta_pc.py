import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np


config_path = Path('../config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

human_accs = pd.read_csv('datasets/human_accs.csv')
model_accs = pd.read_csv('datasets/model_accs_human_subset.csv')
model_accs = model_accs[model_accs["method"] == "head_model"]
task_display_names = {task: config['tasks'][task]['display_name'] for task in config['tasks']}
    
# Build delta_pc dataframe for HUMAN data
data_rows_human = []
for task in human_accs['task'].unique():
    for model_type in config['model_types']:
        # Get baseline mean accuracy across subjects
        baseline_data = human_accs[
            (human_accs['task'] == task) & 
            (human_accs['implant_type'] == "6_15") & 
            (human_accs['model_type'] == model_type)
        ]
        baseline_mean = baseline_data['accuracy'].mean()
        
        for implant_type in config['implant_types']:
            task_data = human_accs[
                (human_accs['task'] == task) & 
                (human_accs['implant_type'] == implant_type) & 
                (human_accs['model_type'] == model_type)
            ]
            
            if len(task_data) > 0:
                # Get accuracies across subjects
                accuracies = task_data['accuracy'].values
                mean_acc = np.mean(accuracies)
                sem_acc = np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))
                
                # Shift so baseline is at 0
                delta_pc = mean_acc - baseline_mean
                
                data_rows_human.append({
                    'task': task,
                    'model_type': model_type,
                    'implant_type': implant_type.replace('_', 'x'),
                    'delta_pc': delta_pc,
                    'sem': sem_acc
                })

df_human = pd.DataFrame(data_rows_human)

# Build delta_pc dataframe for MODEL data
data_rows_model = []
for task in model_accs['task'].unique():
    for model_type in config['model_types']:
        # Get baseline mean accuracy across runs
        baseline_data = model_accs[
            (model_accs['task'] == task) & 
            (model_accs['implant_type'] == "6_15") & 
            (model_accs['model_type'] == model_type)
        ]
        baseline_mean = baseline_data['accuracy'].mean()
        
        for implant_type in config['implant_types']:
            task_data = model_accs[
                (model_accs['task'] == task) & 
                (model_accs['implant_type'] == implant_type) & 
                (model_accs['model_type'] == model_type)
            ]
            
            if len(task_data) > 0:
                # Get accuracies across runs
                accuracies = task_data['accuracy'].values
                mean_acc = np.mean(accuracies)
                sem_acc = np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))
                
                # Shift so baseline is at 0
                delta_pc = mean_acc - baseline_mean
                
                data_rows_model.append({
                    'task': task,
                    'model_type': model_type,
                    'implant_type': implant_type.replace('_', 'x'),
                    'delta_pc': delta_pc,
                    'sem': sem_acc
                })

df_model = pd.DataFrame(data_rows_model)



# Set seaborn style
# Set the style to remove gridlines and keep only left and bottom axes
sns.set_style("ticks")

# Set font size to 7pt (using default font family to match accs_by_task.py)
plt.rcParams.update({'font.size': 7})

# Get unique tasks and create color palette and marker styles
tasks = sorted(df_human['task'].unique())
colors = sns.color_palette("husl", len(tasks))
task_colors = dict(zip(tasks, colors))

# ========== COMBINED 2x2 PLOT ==========
fig, axes = plt.subplots(2, len(config['model_types']), figsize=(6, 5), sharey=True)

# Row 1: Human data
for idx, model_type in enumerate(config['model_types']):
    ax = axes[0, idx]
    model_data = df_human[df_human['model_type'] == model_type]
    
    # Plot lines with markers and error bars for each task
    for task in tasks:
        task_data = model_data[model_data['task'] == task]
        # Sort by the implant_types order from config (only for implant types that exist)
        task_data = task_data.set_index('implant_type')
        available_implants = [imp.replace('_', 'x') for imp in config['implant_types'] if imp.replace('_', 'x') in task_data.index]
        task_data = task_data.loc[available_implants].reset_index()
        # Drop rows with NaN values
        task_data = task_data.dropna(subset=['delta_pc'])
        
        ax.errorbar(task_data['implant_type'], task_data['delta_pc'], 
                    yerr=task_data['sem'],
                    marker="", label=task_display_names[task], color=task_colors[task], 
                    linewidth=2, markersize=8, capsize=3, capthick=1.5, alpha=0.5)
    
    # Formatting
    display_name = config['model_types_display_names'].get(model_type, model_type)
    ax.set_title(f"{display_name} (Humans)")
    if idx == 0:
        ax.set_ylabel('Δ Proportion Correct\n(vs. 6x15 baseline)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(False)
    
    # Rotate x-axis labels if needed
    ax.tick_params(axis='x', rotation=45)

# Row 2: Model data
for idx, model_type in enumerate(config['model_types']):
    ax = axes[1, idx]
    model_data = df_model[df_model['model_type'] == model_type]
    
    # Plot lines with markers and error bars for each task
    for task in tasks:
        task_data = model_data[model_data['task'] == task]
        # Sort by the implant_types order from config (only for implant types that exist)
        task_data = task_data.set_index('implant_type')
        available_implants = [imp.replace('_', 'x') for imp in config['implant_types'] if imp.replace('_', 'x') in task_data.index]
        task_data = task_data.loc[available_implants].reset_index()
        # Drop rows with NaN values
        task_data = task_data.dropna(subset=['delta_pc'])
        
        ax.errorbar(task_data['implant_type'], task_data['delta_pc'], 
                    yerr=task_data['sem'],
                    marker="", label=task_display_names[task], color=task_colors[task], 
                    linewidth=2, markersize=8, capsize=3, capthick=1.5, alpha=0.5)
    
    # Formatting
    display_name = config['model_types_display_names'].get(model_type, model_type)
    ax.set_title(f"{display_name} (CVP)")
    if idx == 0:
        ax.set_ylabel('Δ Proportion Correct\n(vs. 6x15 baseline)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(False)
    
    # Rotate x-axis labels if needed
    ax.tick_params(axis='x', rotation=45)

# Create a single shared horizontal legend positioned between the two rows
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, title='', fontsize=7, loc='center', bbox_to_anchor=(0.5, 0), ncol=len(tasks), frameon=False)

sns.despine(top=True, right=True, fig=fig)
plt.tight_layout()
plt.savefig('plots/delta_pc.pdf', dpi=300, bbox_inches='tight')
print("Plot saved as 'delta_pc.pdf'")
plt.show()



