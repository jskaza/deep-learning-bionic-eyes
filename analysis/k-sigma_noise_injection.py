"""K-Sigma Noise Injection: Optimizes k to minimize MSE between human and model accuracy."""

import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm

np.random.seed(42)
N_NOISE_SAMPLES = 10000
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")
TASKS = ["emotion", "doorway", "shape"]

def compute_noisy_accuracy(preds_df, sigmas_df, k):
    results = []
    for _, row in sigmas_df.iterrows():
        preds = preds_df[(preds_df["implant_type"] == row["implant_type"]) & 
                         (preds_df["model_type"] == row["model_type"])]
        if len(preds) == 0:
            continue
        noise_std = k * row["sigma"]
        run_accs = []
        for run in preds["run"].unique():
            p = preds[preds["run"] == run]
            logits, labels = np.stack(p["logits"].values), p["numeric_label"].values
            if noise_std == 0:
                run_accs.append((np.argmax(logits, axis=1) == labels).mean())
            else:
                noise = np.random.normal(0, noise_std, (N_NOISE_SAMPLES, *logits.shape))
                run_accs.append((np.argmax(logits[None, :, :] + noise, axis=2) == labels).mean())
        results.append({"implant_type": row["implant_type"], "model_type": row["model_type"],
                        "accuracy_with_noise": np.mean(run_accs)})
    return pd.DataFrame(results) if results else None


def compute_mse(human_df, model_df):
    m = human_df.merge(model_df, on=["implant_type", "model_type"], how="inner")
    return np.mean((m["accuracy"] - m["accuracy_with_noise"]) ** 2) if len(m) else None


def optimize_k(preds_df, sigmas_df, human_df):
    def obj(k):
        acc = compute_noisy_accuracy(preds_df, sigmas_df, k)
        return compute_mse(human_df, acc) if acc is not None else np.inf
    return minimize_scalar(obj, bounds=(0, 50), method='bounded', options={'xatol': 0.01})

# Load data
sigmas = pd.read_csv(os.path.join(DATASETS_DIR, "noise_sigmas.csv"))
model_preds = pd.read_pickle(os.path.join(DATASETS_DIR, "model_predictions.pkl"))
model_preds = model_preds[(model_preds["method"] == "full_model") & (model_preds["task"].isin(TASKS))]
human_accs = pd.read_csv(os.path.join(DATASETS_DIR, "human_accs.csv"))
human_accs = human_accs[human_accs["task"].isin(TASKS)]
human_by_cond = human_accs.groupby(["task", "implant_type", "model_type"]).agg(
    accuracy=("accuracy", "mean")).reset_index()

# Optimize k for each task/architecture
results, all_optimized = [], []
for _, combo in tqdm(list(sigmas[["task", "architecture"]].drop_duplicates().iterrows())):
    task, arch = combo["task"], combo["architecture"]
    preds = model_preds[(model_preds["task"] == task) & (model_preds["architecture"] == arch)]
    sigs = sigmas[(sigmas["task"] == task) & (sigmas["architecture"] == arch)]
    human = human_by_cond[human_by_cond["task"] == task]
    if len(preds) == 0 or len(sigs) == 0 or len(human) == 0:
        continue
    
    opt = optimize_k(preds, sigs, human)
    final = compute_noisy_accuracy(preds, sigs, opt.x)
    if final is None:
        continue
    
    final = final.copy()
    final["task"], final["architecture"] = task, arch
    all_optimized.append(final)
    results.append({"task": task, "architecture": arch, "optimal_k": opt.x})

# Create combined results dataframe
results_df = pd.DataFrame(results)
optimized_df = pd.concat(all_optimized, ignore_index=True)

# Merge to get all requested columns
combined_df = optimized_df.merge(
    results_df, 
    on=["task", "architecture"], 
    how="left"
).rename(columns={"accuracy_with_noise": "accuracy"})

# Reorder columns as requested
combined_df = combined_df[["architecture", "optimal_k", "task", "implant_type", "model_type", "accuracy"]]
combined_df = combined_df.sort_values(["task", "architecture", "implant_type", "model_type"]).reset_index(drop=True)

combined_df.to_csv(os.path.join(DATASETS_DIR, "k-sigma_noise_injection_results.csv"), index=False)