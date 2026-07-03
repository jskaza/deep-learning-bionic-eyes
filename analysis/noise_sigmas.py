import pandas as pd
import os
import numpy as np

df = pd.read_pickle(os.path.join("datasets", "model_predictions_human_subset.pkl"))
df = df[df["method"] == "full_model"]
df = df[df["task"].isin(["emotion", "doorway", "shape"])]

def compute_sigma(group):
    """Compute mean of standard deviations across logits for a group."""
    logits = np.stack(group["logits"].values)
    sds = np.std(logits, axis=0)
    return np.mean(sds)

# Group by task, architecture, implant_type, model_type, and target
# Then compute sigma for each combination
result = df.groupby(["task", "architecture", "implant_type", "model_type", "target"]).apply(
    compute_sigma, include_groups=False
).reset_index(name="sigma_per_target")


# Aggregate to get mean sigma across all targets for each task/architecture/implant/model combination
sigmas = result.groupby(["task", "architecture", "implant_type", "model_type"])["sigma_per_target"].mean().reset_index(name="sigma")

sigmas.to_csv(os.path.join("datasets", "noise_sigmas.csv"), index=False)