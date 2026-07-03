import pandas as pd
import numpy as np
import os
from itertools import combinations
from tqdm import tqdm

import numpy as np

def permutation_test(subject_strata_data, observed_jaccard, n_permutations=10000):
    n_subj = len(subject_strata_data)
    weights = np.array([s['weight'] for s in subject_strata_data])

    total_intersection = np.zeros((n_subj, n_permutations))
    total_union = np.zeros((n_subj, n_permutations))

    for subj_idx, subj_data in enumerate(subject_strata_data):
        for block in subj_data['strata_blocks']:
            model_errors = block['model_errors']
            human_errors = block['human_errors']

            n = model_errors.size
            k = int(np.count_nonzero(model_errors))   # # of 1s to be shuffled
            h = int(np.count_nonzero(human_errors))    # fixed "draw" size

            if k == 0 or h == 0:
                inter = np.zeros(n_permutations)
            else:
                inter = np.random.hypergeometric(k, n - k, h, size=n_permutations)

            total_intersection[subj_idx] += inter
            total_union[subj_idx] += (k + h - inter)

    # avoid div-by-zero; union==0 -> jaccard defined as 1.0
    safe_union = np.where(total_union == 0, 1, total_union)
    subj_jaccards = np.where(total_union == 0, 1.0, total_intersection / safe_union)

    null_distribution = np.average(subj_jaccards, axis=0, weights=weights)
    p_value = np.mean(null_distribution >= observed_jaccard)
    return p_value, null_distribution

################################################################################
# Model-Human Similarity Metrics (Jaccard) of Incorrect Predictions 
################################################################################

human_trials = pd.read_csv(os.path.join("datasets", "human_trials.csv"))
model_predictions = pd.read_pickle(os.path.join("datasets", "model_predictions_human_subset.pkl"))
model_predictions = model_predictions[model_predictions["method"].isin(["head_model", "full_model"])]

n_bootstrap = 10000
np.random.seed(42)

subjects = human_trials["subject"].unique()
n_subj = len(subjects)
subj_to_idx = {s: i for i, s in enumerate(subjects)}

# Precompute bootstrap subject index samples (sample with replacement)
boot_indices = np.random.randint(0, n_subj, size=(n_bootstrap, n_subj))

# Compute pairwise human-human metrics as baseline
human_pair_jaccards = {}
for s1, s2 in tqdm(combinations(subjects, 2), total=len(subjects) * (len(subjects) - 1) // 2, desc="Computing human-human metrics"):
    ht1 = human_trials[human_trials["subject"] == s1]
    ht2 = human_trials[human_trials["subject"] == s2]
    merged = ht1.merge(ht2, on=["task", "implant_type", "model_type", "percept_filename"], suffixes=("_1", "_2"))
    common_trials = len(merged)

    if common_trials > 0:
        errors_1 = ~merged["result_1"].values
        errors_2 = ~merged["result_2"].values
        intersection = np.sum(errors_1 & errors_2)
        union = np.sum(errors_1 | errors_2)
        jaccard = 1.0 if union == 0 else intersection / union
        human_pair_jaccards[frozenset([s1, s2])] = (jaccard, common_trials)

js = [v[0] for v in human_pair_jaccards.values()]
ws = [v[1] for v in human_pair_jaccards.values()]
human_jaccard_baseline = np.average(js, weights=ws)

# Build symmetric pair matrices for vectorized bootstrap
num_matrix_hh = np.zeros((n_subj, n_subj))
den_matrix_hh = np.zeros((n_subj, n_subj))
for key, (jaccard, weight) in human_pair_jaccards.items():
    s1, s2 = list(key)
    i, j = subj_to_idx[s1], subj_to_idx[s2]
    num_matrix_hh[i, j] = jaccard * weight
    num_matrix_hh[j, i] = jaccard * weight
    den_matrix_hh[i, j] = weight
    den_matrix_hh[j, i] = weight

# Vectorized bootstrap SE: counts @ M @ counts / 2 gives the weighted pair sum
# (exploiting symmetry of M and zero diagonal to avoid explicit pair enumeration)
counts = np.vstack([np.bincount(bi, minlength=n_subj) for bi in boot_indices]).astype(float)
total_num = ((counts @ num_matrix_hh) * counts).sum(axis=1) / 2
total_den = ((counts @ den_matrix_hh) * counts).sum(axis=1) / 2
has_data = total_den > 0
human_jaccard_se = np.std(total_num[has_data] / total_den[has_data])

results = [{
    "architecture": "human",
    "method": "human",
    "jaccard": human_jaccard_baseline,
    "se": human_jaccard_se,
    "p": np.nan,
}]

for architecture in tqdm(model_predictions["architecture"].unique(), desc="Computing model-human metrics"):
    for method in model_predictions["method"].unique():
        df_m = model_predictions[(model_predictions["architecture"] == architecture) & (model_predictions["method"] == method)]

        # Organize data by subject for bootstrap SE, observed Jaccard, and permutation test.
        # Each subject entry stores their overall Jaccard + per-stratum blocks so that
        # the permutation test can shuffle model errors within strata independently per subject.
        subject_data = {}
        subject_strata_data = []

        for subject in subjects:
            human_trials_subject = human_trials[human_trials["subject"] == subject]
            df_merged = df_m.merge(human_trials_subject, on=["task", "implant_type", "model_type", "percept_filename"], how="inner", suffixes=("_model", "_human"))

            if len(df_merged) == 0:
                continue

            model_errors = ~df_merged["result_model"].values
            human_errors = ~df_merged["result_human"].values
            intersection = np.sum(model_errors & human_errors)
            union = np.sum(model_errors | human_errors)
            jaccard = 1.0 if union == 0 else intersection / union

            subject_data[subject] = [{
                'jaccard': jaccard,
                'weight': len(df_merged),
            }]

            # Build per-stratum blocks for this subject (used by the permutation test)
            strata_blocks = [
                {
                    'model_errors': ~grp["result_model"].values,
                    'human_errors': ~grp["result_human"].values,
                }
                for _, grp in df_merged.groupby(["task", "run_model", "implant_type", "model_type"])
            ]

            subject_strata_data.append({
                'strata_blocks': strata_blocks,
                'weight': len(df_merged),
            })
        
        
        # Observed weighted mean Jaccard
        all_js = [e['jaccard'] for entries in subject_data.values() for e in entries]
        all_ws = [e['weight'] for entries in subject_data.values() for e in entries]
        mean_jaccard = np.average(all_js, weights=all_ws)

        # Vectorized bootstrap SE: precompute per-subject weighted sums, then
        # use fancy indexing to gather and sum across all bootstraps at once
        subj_numerators = np.zeros(n_subj)
        subj_denominators = np.zeros(n_subj)
        for subject, entries in subject_data.items():
            idx = subj_to_idx[subject]
            for entry in entries:
                subj_numerators[idx] += entry['jaccard'] * entry['weight']
                subj_denominators[idx] += entry['weight']

        boot_nums = subj_numerators[boot_indices].sum(axis=1)
        boot_dens = subj_denominators[boot_indices].sum(axis=1)
        has_data = boot_dens > 0
        jaccard_se = np.std(boot_nums[has_data] / boot_dens[has_data])

        # Permutation test: shuffle model errors within (task, run, implant_type, model_type)
        # strata independently per subject, then weight-average per-subject Jaccards
        p_value_jaccard, null_dist_jaccard = permutation_test(
            subject_strata_data, mean_jaccard
        )

        results.append({
            "architecture": architecture,
            "method": method,
            "jaccard": mean_jaccard,
            "se": jaccard_se,
            "p": p_value_jaccard,
        })

results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("JACCARD INDEX RESULTS")
print("="*80)
print(results_df[["architecture", "method", "jaccard", "se", "p"]].sort_values(by=["method", "jaccard"]).to_markdown())

results_df.to_csv(os.path.join("datasets", "model_human_jaccard.csv"), index=False)
