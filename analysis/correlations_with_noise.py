import pandas as pd
import os
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
import json
from scipy.stats import false_discovery_control

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Chance level = 1 / num_classes; used to normalize accuracy_above_chance
chance_levels = {
    task: 1.0 / len(config['tasks'][task]['labels'])
    for task in config['tasks']
}

################################################################################
# Model-Human Correlation of Accuracy Above Chance
# Computes Pearson r between model and human accuracy-above-chance,
# across all (task, implant_type, model_type) configurations.
################################################################################

m_base = pd.read_csv(os.path.join("datasets", "model_accs_human_subset.csv"))
h = pd.read_csv(os.path.join("datasets", "human_accs.csv"))

# Average human accuracy over subjects for each experimental configuration
h_by_config = h.groupby(["task", "implant_type", "model_type"]).agg(
    accuracy_above_chance=("accuracy_above_chance", "mean"),
).reset_index()

# Average model accuracy over seeds/runs for each configuration
m_by_config = m_base.groupby(["architecture", "method", "task", "implant_type", "model_type"]).agg(
    accuracy_above_chance=("accuracy_above_chance", "mean"),
).reset_index()
m_by_config["method"] = m_by_config["method"].replace({"head_model": "Linear Probe", "full_model": "End-to-End"})

# Noise injection baselines: accuracy_above_chance = (acc - chance) / (1 - chance),
# normalising so that 0 = chance performance and 1 = perfect accuracy.
m_additive = pd.read_csv(os.path.join("datasets", "additive_noise_injection_results.csv"))
m_additive["method"] = "Additive Noise"
m_additive["accuracy_above_chance"] = (
    (m_additive["accuracy"] - m_additive["task"].map(chance_levels))
    / (1 - m_additive["task"].map(chance_levels))
)

m_k_sigma = pd.read_csv(os.path.join("datasets", "k-sigma_noise_injection_results.csv"))
m_k_sigma["accuracy_above_chance"] = (
    (m_k_sigma["accuracy"] - m_k_sigma["task"].map(chance_levels))
    / (1 - m_k_sigma["task"].map(chance_levels))
)
m_k_sigma["method"] = "Proportional Noise"

m_by_config = pd.concat([m_by_config, m_additive, m_k_sigma])


df_by_config = m_by_config.merge(h_by_config, on=["task", "implant_type", "model_type"], how="inner", suffixes=("_model", "_human"))

n_bootstrap = 10000

# Bootstrap human data by resampling subjects (runs) with replacement within each task.
# Stratifying by task preserves the task distribution while estimating sampling variability.
np.random.seed(42)
bootstrapped_h_samples = []
for _ in tqdm(range(n_bootstrap), desc="Bootstrapping human samples"):
    boot_dfs = []
    for task, group in h.groupby("task"):
        subjects = group["run"].unique()
        sampled_subjects = np.random.choice(subjects, size=len(subjects), replace=True)
        boot_dfs.append(pd.concat([group[group["run"] == s] for s in sampled_subjects]))
    bootstrapped_h_samples.append(pd.concat(boot_dfs))

# Accumulator: bootstrapped_rs[arch][method] = list of r values across bootstrap samples
bootstrapped_rs = {arch: {method: [] for method in df_by_config["method"].unique()} for arch in df_by_config["architecture"].unique()}

for arch, group in tqdm(m_by_config.groupby("architecture"), total=len(m_by_config["architecture"].unique()), desc="Computing model-human correlations for each architecture"):
    for method in group["method"].unique():
        model_acc = group[group["method"] == method]
        for boot_h in bootstrapped_h_samples:
            # Average bootstrapped human data to the same (task, implant_type, model_type) grain as model data
            boot_h_by_config = boot_h.groupby(["task", "implant_type", "model_type"]).agg(
                accuracy_above_chance=("accuracy_above_chance", "mean"),
            ).reset_index()
            _df = pd.merge(model_acc, boot_h_by_config, on=["task", "implant_type", "model_type"], how="inner", suffixes=("_model", "_human"))
            boot_r, _ = pearsonr(_df["accuracy_above_chance_model"], _df["accuracy_above_chance_human"])
            bootstrapped_rs[arch][method].append(boot_r)

# Point-estimate correlations (no bootstrapping) for each architecture-method pair
base_rs = []
for arch, group in m_by_config.groupby("architecture"):
    for method in group["method"].unique():
        model_acc = group[group["method"] == method]
        _df = pd.merge(model_acc, h_by_config, on=["task", "implant_type", "model_type"], how="inner", suffixes=("_model", "_human"))
        r, p = pearsonr(_df["accuracy_above_chance_model"], _df["accuracy_above_chance_human"])
        base_rs.append({"architecture": arch, "method": method, "r": r, "p": p})

base_rs_df = pd.DataFrame(base_rs)
base_rs_df.rename(columns={"architecture": "Architecture"}, inplace=True)
print("\nBase correlations:")
pivot = base_rs_df.pivot(index="Architecture", columns="method", values="r")
pivot.loc["Overall"] = pivot.mean()
pivot["Architecture"] = pivot.index

print(pivot[["Architecture", "Linear Probe", "End-to-End", "Additive Noise", "Proportional Noise"]].to_latex(index=False, float_format="%.3f"))

# Bootstrap p-values: proportion of bootstrap samples where method A correlation < method B correlation.
# Tests whether noise-injection baselines outperform the Linear Probe.
ps = []
for arch in bootstrapped_rs.keys():
    head = np.array(bootstrapped_rs[arch]["Linear Probe"])
    full = np.array(bootstrapped_rs[arch]["End-to-End"])
    k_sigma = np.array(bootstrapped_rs[arch]["Proportional Noise"])
    additive = np.array(bootstrapped_rs[arch]["Additive Noise"])
    ps.extend([
        {"arch": arch, "comparison": "linear probe < end-to-end",                    "p": np.mean(head < full)},
        {"arch": arch, "comparison": "linear probe < proportional noise injection",   "p": np.mean(head < k_sigma)},
        {"arch": arch, "comparison": "linear probe < additive noise injection",       "p": np.mean(head < additive)},
    ])

ps_df = pd.DataFrame(ps)
# Benjamini-Hochberg FDR correction across per-architecture comparisons only
ps_df["adjusted_p"] = false_discovery_control(ps_df["p"])
pivot = ps_df.pivot(index="arch", columns="comparison", values="adjusted_p")
pivot["Architecture"] = pivot.index

# Append overall (architecture-pooled) p-values as a summary row
all_head     = np.vstack([bootstrapped_rs[arch]["Linear Probe"]      for arch in bootstrapped_rs.keys()]).mean(axis=0)
all_full     = np.vstack([bootstrapped_rs[arch]["End-to-End"]         for arch in bootstrapped_rs.keys()]).mean(axis=0)
all_k_sigma  = np.vstack([bootstrapped_rs[arch]["Proportional Noise"] for arch in bootstrapped_rs.keys()]).mean(axis=0)
all_additive = np.vstack([bootstrapped_rs[arch]["Additive Noise"]     for arch in bootstrapped_rs.keys()]).mean(axis=0)
pivot.loc["Overall", "linear probe < end-to-end"]                    = np.mean(all_head < all_full)
pivot.loc["Overall", "linear probe < proportional noise injection"]   = np.mean(all_head < all_k_sigma)
pivot.loc["Overall", "linear probe < additive noise injection"]       = np.mean(all_head < all_additive)
pivot.loc["Overall", "Architecture"] = "Overall"

fmt_p = lambda x: r"$<$0.001" if x < 0.001 else f"{x:.3f}"
print(pivot[["Architecture", "linear probe < end-to-end", "linear probe < additive noise injection", "linear probe < proportional noise injection"]].to_latex(index=False, float_format=fmt_p))


