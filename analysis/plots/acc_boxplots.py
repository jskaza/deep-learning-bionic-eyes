import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
from pathlib import Path

# Set the style to remove gridlines and keep only left and bottom axes
sns.set_style("ticks")
sns.despine(top=True, right=True)

config_path = Path('../../config.json')
with config_path.open('r') as f:
    config = json.load(f)
chance_lines = {}
task_display_names = {}
for task, info in config['tasks'].items():
    chance_lines[task] = 1/len(info["labels"])
    task_display_names[task] = info["display_name"]

##### Fig 1: Accuracy by task
# Read the data
df = pd.read_csv('../summary_datasets/accuracy_by_task_implant_model_architecture_run.csv')

# Create a new column to separate human and non-human architectures
df['architecture_type'] = df['architecture'].apply(lambda x: 'Human' if x == 'human' else 'DNN')

# Calculate median accuracies for each task
task_order = df.groupby('task')['accuracy'].median().sort_values(ascending=True).index.tolist()

# Set seaborn style
sns.set_style("ticks")
plt.figure(figsize=(10, 4))

# Set custom colors for Human and DNN
custom_palette = {'Human': '#f15a29', 'DNN': '#00aeef'}

# Create boxplot with hue to separate human and non-human architectures
sns.boxplot(data=df, x='task', y='accuracy', hue='architecture_type', order=task_order, 
            hue_order=['Human', 'DNN'], palette=custom_palette, showfliers=False, legend=False)

# Add vertical lines between tasks
for i in range(len(task_order)-1):
    plt.axvline(x=i+0.5, color='gray', linestyle='-', alpha=0.3)

# Format x-axis tick labels using display names from config
ax = plt.gca()
ax.set_xticks(range(len(task_order)))
ax.set_xticklabels([task_display_names[task] for task in task_order])

# Create marker mapping for model_type
marker_map = {'streaky': 'o', 'pointy': 's'}

size_map = {'6_10' : 60/60,
        "9_10" : 90/60,
        "6_15" : 90/60,
        "12_10" : 120/60,
        "6_20" : 120/60,
        "12_20" : 240/60}

# Add jittered points with different markers and sizes
for model_type in ['streaky', 'pointy']:
    for implant_type, size in size_map.items():
        subset = df[(df['model_type'] == model_type) & (df['implant_type'] == implant_type)]
        if not subset.empty:
            sns.stripplot(data=subset, x='task', y='accuracy', hue='architecture_type', order=task_order,
                      hue_order=['Human', 'DNN'], palette=custom_palette, dodge=True, 
                      marker=marker_map[model_type], size=size, alpha=0.3, legend=False)

# Add chance lines for each task
for i, task in enumerate(task_order):
    plt.axhline(y=chance_lines[task], xmin=i/len(task_order), xmax=(i+1)/len(task_order), 
                color='gray', linestyle='--', alpha=0.5)

# Remove top and right spines
sns.despine(top=True, right=True)

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.xlabel('')
plt.ylabel('Proportion Correct')
# Save the plot
plt.savefig('accuracy_by_task_boxplot.pdf')
plt.close()
















