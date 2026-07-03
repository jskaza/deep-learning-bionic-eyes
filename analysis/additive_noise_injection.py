"""Additive Noise Injection: Optimizes sigma to minimize MSE between human and model accuracy."""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm

np.random.seed(42)
N_NOISE_SAMPLES = 10000
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")
TASKS = ["emotion", "doorway", "shape"]

def compute_noisy_accuracy(preds_df, sigma):
    """Compute accuracy with additive Gaussian noise N(0, sigma) on logits."""
    results = []
    for (it, mt), group in preds_df.groupby(["implant_type", "model_type"]):
        run_accs = []
        for run in group["run"].unique():
            p = group[group["run"] == run]
            logits, labels = np.stack(p["logits"].values), p["numeric_label"].values
            if sigma == 0:
                run_accs.append((np.argmax(logits, axis=1) == labels).mean())
            else:
                noise = np.random.normal(0, sigma, (N_NOISE_SAMPLES, *logits.shape))
                run_accs.append((np.argmax(logits[None, :, :] + noise, axis=2) == labels).mean())
        results.append({"implant_type": it, "model_type": mt,
                        "accuracy_with_noise": np.mean(run_accs)})
    return pd.DataFrame(results) if results else None

def compute_mse(human_df, model_df):
    m = human_df.merge(model_df, on=["implant_type", "model_type"], how="inner")
    return np.mean((m["accuracy"] - m["accuracy_with_noise"]) ** 2) if len(m) else None

def optimize_sigma(preds_df, human_df):
    """Optimize sigma directly to minimize MSE between human and model accuracy."""
    def obj(sigma):
        acc = compute_noisy_accuracy(preds_df, sigma)
        return compute_mse(human_df, acc) if acc is not None else np.inf
    return minimize_scalar(obj, bounds=(0, 50), method='bounded', options={'xatol': 0.01})

# Load data
model_preds = pd.read_pickle(os.path.join(DATASETS_DIR, "model_predictions_human_subset.pkl"))
model_preds = model_preds[(model_preds["method"] == "full_model") & (model_preds["task"].isin(TASKS))]
human_accs = pd.read_csv(os.path.join(DATASETS_DIR, "human_accs.csv"))
human_accs = human_accs[human_accs["task"].isin(TASKS)]
human_by_cond = human_accs.groupby(["task", "implant_type", "model_type"]).agg(
    accuracy=("accuracy", "mean")).reset_index()

# Optimize sigma for each task/architecture
results, all_optimized = [], []
for task in tqdm(TASKS):
    for arch in tqdm(model_preds[model_preds["task"] == task]["architecture"].unique()):
        preds = model_preds[(model_preds["task"] == task) & (model_preds["architecture"] == arch)]
        human = human_by_cond[human_by_cond["task"] == task]
        if len(preds) == 0 or len(human) == 0:
            continue
        
        opt = optimize_sigma(preds, human)
        final = compute_noisy_accuracy(preds, opt.x)
        if final is None:
            continue
        
        final = final.copy()
        final["task"], final["architecture"] = task, arch
        all_optimized.append(final)
        results.append({"task": task, "architecture": arch, "optimal_sigma": opt.x})

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
combined_df = combined_df[["architecture", "optimal_sigma", "task", "implant_type", "model_type", "accuracy"]]
combined_df = combined_df.sort_values(["task", "architecture", "implant_type", "model_type"]).reset_index(drop=True)

combined_df.to_csv(os.path.join(DATASETS_DIR, "additive_noise_injection_results.csv"), index=False)
