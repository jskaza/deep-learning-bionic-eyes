import pandas as pd
import os
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm

################################################################################
# Model-Human Correlation of Accuracy Above Chance 
# across tasks, implant types, and model types
################################################################################

m_base = pd.read_csv(os.path.join("datasets", "model_accs_human_subset.csv"))
h = pd.read_csv(os.path.join("datasets", "human_accs.csv"))
h_by_config = h.groupby(["task", "implant_type", "model_type"]).agg(
    accuracy_above_chance=("accuracy_above_chance", "mean"),
).reset_index()

m_by_config = m_base.groupby(["architecture", "method", "task", "implant_type", "model_type"]).agg(
    accuracy_above_chance=("accuracy_above_chance", "mean"),
).reset_index()

n_bootstrap = 10000
np.random.seed(42)

# Precompute bootstrapped human dfs (sample subjects with replacement)
bootstrapped_h_samples = []
for _ in tqdm(range(n_bootstrap), desc="Bootstrapping human samples"):
    boot_dfs = []
    for task, group in h.groupby("task"):
        subjects = group["run"].unique()
        sampled_subjects = np.random.choice(subjects, size=len(subjects), replace=True)
        boot_dfs.append(pd.concat([group[group["run"] == s] for s in sampled_subjects]))
    bootstrapped_h_samples.append(pd.concat(boot_dfs))

# Base correlation + bootstrapped SE for each architecture-method
results = []
for arch, arch_group in tqdm(m_by_config.groupby("architecture"), total=m_by_config["architecture"].nunique(), desc="Computing correlations"):
    for method, method_group in arch_group.groupby("method"):
        _df = pd.merge(method_group, h_by_config, on=["task", "implant_type", "model_type"], how="inner", suffixes=("_model", "_human"))
        r, p = pearsonr(_df["accuracy_above_chance_model"], _df["accuracy_above_chance_human"])

        boot_rs = []
        for boot_h in bootstrapped_h_samples:
            boot_h_by_config = boot_h.groupby(["task", "implant_type", "model_type"]).agg(
                accuracy_above_chance=("accuracy_above_chance", "mean"),
            ).reset_index()
            _df_boot = pd.merge(method_group, boot_h_by_config, on=["task", "implant_type", "model_type"], how="inner", suffixes=("_model", "_human"))
            boot_r, _ = pearsonr(_df_boot["accuracy_above_chance_model"], _df_boot["accuracy_above_chance_human"])
            boot_rs.append(boot_r)

        results.append({"architecture": arch, "method": method, "r": r, "p": p, "se": np.std(boot_rs)})

results_df = pd.DataFrame(results)
print("\nCorrelations with bootstrapped SE:")
print(results_df.sort_values(by=["method","r"]).to_markdown(index=False))

results_df.to_csv(os.path.join("datasets", "model_human_correlations_head_full.csv"), index=False)
