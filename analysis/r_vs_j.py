import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

annotate = False
sns.set_style("ticks")

# Set font size to 7pt (using default font family to match accs_by_task.py)
plt.rcParams.update({'font.size': 7})

r_df = pd.read_csv(os.path.join("datasets", "model_human_correlations_head_full.csv"))
j_df = pd.read_csv(os.path.join("datasets", "model_human_jaccard.csv"))

df = r_df.merge(j_df, on=["architecture", "method"], how="inner", suffixes=("_r", "_j"))
# Create scatter plot with error bars
fig, ax = plt.subplots(figsize=(6, 3))

# Get unique methods for coloring
methods = df['method'].unique()
colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
method_colors = dict(zip(methods, colors))

# Plot each point with error bars
for _, row in df.iterrows():
    method = row['method']
    color = method_colors[method]
    
    if annotate:
        ax.annotate(
            row['architecture'],
            xy=(row['jaccard'], row['r']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=5,
            alpha=0.8,
            color=color
        )
    ax.errorbar(
        row['jaccard'],
        row['r'],
        xerr=row['se_j'],
        yerr=row['se_r'],
        fmt='o',
        color=color,
        alpha=0.7,
        capsize=3,
        capthick=1.5,
        elinewidth=1.5
    )

# Add labels and title
ax.set_xlabel('Misclassification Overlap (Jaccard Index)')
ax.set_ylabel('Correlation between CVP and humans \n(Pearson r)')

sns.despine(top=True, right=True, fig=fig)


plt.tight_layout()
plt.savefig(os.path.join("plots", "r_vs_jaccard_scatter.pdf"), bbox_inches='tight')
plt.show()


df.rename(
    columns={
        "architecture": "Architecture",
        "method": "Method",
        "r": "Pearson r",
        "jaccard": "Jaccard Index (J)",
        "se_r": "SE (r)",
        "se_j": "SE (J)",
        "p_r": "p (r)",
        "p_j": "p (J)",
    },
    inplace=True,
)
df["Method"] = df["Method"].replace({"head_model": "Linear Probe", "full_model": "End-to-End"})

df_sorted = df.sort_values(by=["Pearson r", "Jaccard Index (J)"], ascending=False)

numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns
p_cols = ["p (r)", "p (J)"]


def format_sigfig(x):
    if pd.isna(x):
        return ""
    return f"{x:.3f}"


def format_p(x):
    if pd.isna(x):
        return ""
    if x < 0.001:
        return "<0.001"
    return f"{x:.3g}"


for col in numeric_cols:
    if col in p_cols:
        df_sorted[col] = df_sorted[col].apply(format_p)
    else:
        df_sorted[col] = df_sorted[col].apply(format_sigfig)

print(
    df_sorted[
        [
            "Architecture",
            "Method",
            "Pearson r",
            "SE (r)",
            "p (r)",
            "Jaccard Index (J)",
            "SE (J)",
            "p (J)",
        ]
    ].to_latex(index=False)
)
