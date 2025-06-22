import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from numpy.random import default_rng
import scipy.stats

def analyze_correlations(pivot_df: pd.DataFrame, human_config_df: pd.Series, 
                        random_seed: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Analyze correlations between architecture families and human data.
    
    Args:
        pivot_df: Pivot table with architecture families as columns
        human_config_df: Series containing human performance data
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing correlation results for each architecture, including percentile CI
    """
    correlation_results = {}
    
    for arch in pivot_df.columns:
        arch_data = pivot_df[arch]
        
        # Align the data by index
        aligned_data = pd.DataFrame({
            'arch': arch_data,
            'human': human_config_df
        }).dropna()
        
        # Calculate correlation with human performance
        correlation_spearman = scipy.stats.spearmanr(aligned_data['arch'], aligned_data['human'])[0]
        correlation_pearson = scipy.stats.pearsonr(aligned_data['arch'], aligned_data['human'])[0]

        correlation_results[arch] = {
            'spearman_correlation': correlation_spearman,
            'pearson_correlation': correlation_pearson
        }
    
    return correlation_results

def create_accuracy_plot(mean_df: pd.DataFrame, config: Dict, 
                        output_path: Path, correlation_results: Dict[str, Dict[str, float]] = None,
                        random_seed: int = 42) -> None:
    """
    Create and save the accuracy plot.
    
    Args:
        mean_df: DataFrame containing mean accuracies and CIs
        config: Configuration dictionary
        output_path: Path to save the plot
        correlation_results: Dictionary containing correlation results for each architecture
        random_seed: Random seed for reproducibility
    """
    # Set the style to remove gridlines and keep only left and bottom axes
    sns.set_style("ticks")
    sns.despine(top=True, right=True)
    
    # Get unique values for subplot layout
    model_types = mean_df['model_type'].unique()
    tasks = ["emotion", "shape", "doorway"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(10, 4))
    
    # Define custom color palette with more prominent human color
    custom_palette = {'human': '#FF4500'}  # Vibrant orange-red
    
    # Create a distinct color palette for architecture families
    blue_palette = {
        'ConvNeXt': '#1f77b4',      # Blue
        'EfficientNet': '#2ca02c',  # Green
        'MobileNetV2': '#9467bd',   # Purple
        'RegNetY': '#d62728',       # Red
        'ResNet': '#bcbd22',        # Olive
        'VGG': '#8c564b',           # Brown
        'Vision Transformer (ViT)': '#17becf'  # Cyan
    }
    
    # Define markers for each architecture
    marker_dict = {
        'ConvNeXt': 'o',           # Circle
        'EfficientNet': 's',       # Square
        'MobileNetV2': '^',        # Triangle up
        'RegNetY': 'D',           # Diamond
        'ResNet': 'v',            # Triangle down
        'VGG': '<',               # Triangle left
        'Vision Transformer (ViT)': '>'  # Triangle right
    }
    
    # Add architecture families to the custom palette
    for d in mean_df['architecture_family'].unique():
        if d != 'human':
            if d in blue_palette:
                custom_palette[d] = blue_palette[d]
            else:
                # Default to a neutral color if architecture not in our palette
                custom_palette[d] = '#808080'
    
    # Plot each subplot
    for i, task in enumerate(tasks):
        for j, model_type in enumerate(model_types):
            ax = axes[i, j]
            
            # Filter data for this subplot
            plot_data = mean_df[(mean_df['task'] == task) & (mean_df['model_type'] == model_type)]
            
            # Special case for emotion task - only plot specific implant types
            if task == "emotion":
                # Filter to only include the implant types we want to show
                plot_data = plot_data[plot_data['implant_type'].isin(['6_15', '12_20'])]
                
                # Create a mapping for x-axis positions based on the full implant_types list
                x_pos_map = {implant: idx for idx, implant in enumerate(config['implant_types'])}
                plot_data = plot_data.assign(x_pos=plot_data['implant_type'].map(x_pos_map))
                
                # Create line plot with different colors for each architecture family
                alpha_dict = {arch: 1.0 if arch == 'human' else 0.4 
                            for arch in plot_data['architecture_family'].unique()}
                g = sns.lineplot(data=plot_data, x='x_pos', y='accuracy', 
                            hue='architecture_family', marker='o', ax=ax, legend=False,
                            palette=custom_palette, linewidth=1)
                
                # Set alpha values, line properties, and markers for each line
                for line, arch in zip(g.lines, plot_data['architecture_family'].unique()):
                    line.set_alpha(alpha_dict[arch])
                    if arch == 'human':
                        line.set_linewidth(2)  # Thicker line for human
                        line.set_markersize(6)   # Larger markers for human
                    else:
                        line.set_marker(marker_dict[arch])  # Set different marker for each architecture
                        line.set_markersize(5)   # Slightly smaller markers for models
                
                # Add error bars for confidence intervals using x_pos
                for arch in plot_data['architecture_family'].unique():
                    arch_data = plot_data[plot_data['architecture_family'] == arch]
                    yerr_lower = arch_data['accuracy'] - arch_data['ci_lower']
                    yerr_upper = arch_data['ci_upper'] - arch_data['accuracy']
                    ax.errorbar(x=arch_data['x_pos'], y=arch_data['accuracy'], 
                               yerr=[yerr_lower, yerr_upper],
                               fmt='none', ecolor=custom_palette[arch], 
                               alpha=alpha_dict[arch], capsize=0, 
                               elinewidth=2 if arch == 'human' else 1)
                               
            else:
                # Create line plot with different colors for each architecture family
                alpha_dict = {arch: 1.0 if arch == 'human' else 0.4 
                            for arch in plot_data['architecture_family'].unique()}
                g = sns.lineplot(data=plot_data, x='implant_type', y='accuracy', 
                            hue='architecture_family', marker='o', ax=ax, legend=False,
                            palette=custom_palette, linewidth=1)
                
                # Set alpha values, line properties, and markers for each line
                for line, arch in zip(g.lines, plot_data['architecture_family'].unique()):
                    line.set_alpha(alpha_dict[arch])
                    if arch == 'human':
                        line.set_linewidth(2)  # Thicker line for human
                        line.set_markersize(6)   # Larger markers for human
                    else:
                        line.set_marker(marker_dict[arch])  # Set different marker for each architecture
                        line.set_markersize(5)   # Slightly smaller markers for models
                
                # Add error bars for confidence intervals using x_pos
                for arch in plot_data['architecture_family'].unique():
                    arch_data = plot_data[plot_data['architecture_family'] == arch]
                    yerr_lower = arch_data['accuracy'] - arch_data['ci_lower']
                    yerr_upper = arch_data['ci_upper'] - arch_data['accuracy']
                    x_pos = [config['implant_types'].index(x) for x in arch_data['implant_type']]
                    ax.errorbar(x=x_pos, y=arch_data['accuracy'], 
                               yerr=[yerr_lower, yerr_upper],
                               fmt='none', ecolor=custom_palette[arch], 
                               alpha=alpha_dict[arch], capsize=0, 
                               elinewidth=2 if arch == 'human' else 1)
            
            # Add chance line
            ax.axhline(y=1/len(config['tasks'][task]["labels"]), color='gray', linestyle='--', alpha=0.5)
            
            # Set labels and title
            if i == 0:  # Only add model type to top row
                model_type_display = config['model_types_display_names'][model_type]
                ax.set_title(model_type_display)
            if j == 0:  # Only add task to leftmost column
                task_display_name = config['tasks'][task]['display_name']
                ax.set_ylabel(f'{task_display_name}\nProportion Correct')
            else:
                ax.set_ylabel('')
            
            # Set y-axis limits and ticks
            ax.set_ylim(0.3, 1)
            ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            
            # Set x-axis ticks and labels properly with underscore replaced by 'x'
            ax.set_xticks(range(len(config['implant_types'])))
            ax.set_xticklabels([label.replace('_', 'x') for label in config['implant_types']])
            ax.set_xlabel('')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    

    
    # Create a legend for architecture families
    handles = []
    labels = []
    for arch, color in custom_palette.items():
        if arch == 'human':
            handles.append(plt.Line2D([0], [0], color=color, marker='o', 
                                    lw=3.5, markersize=8))
            labels.append(arch.capitalize())
        else:
            handles.append(plt.Line2D([0], [0], color=color, marker=marker_dict[arch], 
                                    lw=2.5, markersize=6))
            if correlation_results and arch in correlation_results:
                spearman = correlation_results[arch]['spearman_correlation']
                labels.append(f"{arch} (ρ={spearman:.2f})")
            else:
                labels.append(arch)
    
    # Add legend to the figure
    fig.legend(handles, labels, 
              loc='upper center',
              bbox_to_anchor=(0.5, -0.05),
              ncol=4,
              title='Architecture Family')
    
    # Save the plot with extra space for legend
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()

def bootstrap_median_ci(data: np.ndarray, n_bootstrap: int = 1000, random_seed: int = 42, ci: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrapped confidence interval for the median using the percentile method.
    
    Args:
        data: Input data array
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed for reproducibility
        ci: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple containing (lower_bound, upper_bound) of the confidence interval
    """
    rng = default_rng(random_seed)
    bootstrap_indices = rng.integers(0, len(data), size=(n_bootstrap, len(data)))
    bootstrap_medians = np.median(data[bootstrap_indices], axis=1)
    lower = np.percentile(bootstrap_medians, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_medians, (1 + ci) / 2 * 100)
    return lower, upper

def main():
    """Main function to run the analysis and create plots."""
    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    
    # Load configuration
    config_path = Path('../../config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Read the data
    df = pd.read_csv('../summary_datasets/accuracy_by_task_implant_model_architecture_run.csv')
    
    # Convert implant_type to categorical with correct order
    df = df.assign(implant_type=pd.Categorical(df['implant_type'], 
                                             categories=config['implant_types'], 
                                             ordered=True))
    
    
    # Create a pivot table of model performance by architecture and task configuration
    model_df = df[df['architecture_family'] != "human"].copy()
    
    # Calculate mean accuracy for each architecture family and configuration
    config_corr_df = model_df.groupby(['architecture_family', 'task', 'implant_type', 'model_type'], observed=True)['accuracy'].median().reset_index()
    
    # Create a pivot table with architecture families as columns and configurations as rows
    pivot_df = config_corr_df.pivot_table(
        index=['task', 'implant_type', 'model_type'], 
        columns='architecture_family', 
        values='accuracy',
        observed=True
    )
    
    # Calculate how each architecture family correlates with the human data
    human_config_df = df[df['architecture_family'] == "human"].groupby(['task', 'implant_type', 'model_type'], observed=True)['accuracy'].median()
    
    # Analyze correlations and save results
    correlation_results = analyze_correlations(pivot_df, human_config_df)
    
    # Save correlation results to JSON
    output_path = Path('correlations.json')
    with open(output_path, 'w') as f:
        json.dump(correlation_results, f, indent=4)
    
    # Calculate median accuracy and bootstrap SE for each group
    result_rows = []
    for (model, task, implant, arch), group in df.groupby(['model_type', 'task', 'implant_type', 'architecture_family'], observed=True):
        accuracies = group['accuracy'].values
        median_acc = np.median(accuracies)
        lower, upper = bootstrap_median_ci(accuracies, random_seed=random_seed)
        result_rows.append({
            'model_type': model,
            'task': task,
            'implant_type': implant,
            'architecture_family': arch,
            'accuracy': median_acc,
            'ci_lower': lower,
            'ci_upper': upper
        })
    
    mean_df = pd.DataFrame(result_rows)
    
    # Create and save the plot
    output_path = Path('accuracy_by_config.pdf')
    create_accuracy_plot(mean_df, config, output_path, correlation_results, random_seed)

if __name__ == "__main__":
    main()
















