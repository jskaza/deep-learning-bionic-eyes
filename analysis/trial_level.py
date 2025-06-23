import numpy as np
from typing import Dict, Tuple
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')  # Set the backend to non-interactive 'Agg'


# Set font size
mpl.rcParams['font.size'] = 7

# Set the style to remove gridlines and keep only left and bottom axes
sns.set_style("ticks")
sns.despine(top=True, right=True)

def _run_bootstrap_analysis(
    stim2A: Dict[str, np.ndarray],
    stim2B: Dict[str, np.ndarray],
    n_bootstrap: int = 1_000,
    correct_threshold: float = 0.5,
    rng: np.random.Generator = None
) -> Tuple[float, Tuple[float, float], float, Tuple[float, float], np.ndarray, np.ndarray]:
    """Compute F1 and Jaccard metrics between two stimulus groups using bootstrap analysis.
    
    Performs bootstrap analysis by:
    1. Computing consensus votes within each stimulus using a threshold
    2. Performing stimulus-level bootstrap sampling
    3. Calculating F1 scores and Jaccard indices for misclassification sets
    
    Args:
        stim2A: Dictionary mapping stimulus IDs to binary result arrays
        stim2B: Dictionary mapping stimulus IDs to binary result arrays
        n_bootstrap: Number of bootstrap iterations
        correct_threshold: Threshold for majority vote (proportion of correct responses)
        rng: Random number generator
    
    Returns:
        Tuple of (mean_f1, (f1_lo, f1_hi), mean_jacc, (jacc_lo, jacc_hi), f1s_raw, jacc_raw) where:
        - mean_f1: Mean F1 score across iterations
        - (f1_lo, f1_hi): 95% confidence interval for F1
        - mean_jacc: Mean Jaccard index across iterations
        - (jacc_lo, jacc_hi): 95% confidence interval for Jaccard
        - f1s_raw: Raw F1 scores from all bootstrap iterations
        - jacc_raw: Raw Jaccard scores from valid bootstrap iterations (may be shorter than n_bootstrap)
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    stims = [
        s for s in stim2A
        if s in stim2B
        and len(stim2A[s]) >= 5
        and len(stim2B[s]) >= 5
    ]
    n_stim = len(stims)

    # Within-stim bootstrap → consensus votes
    M_A = np.empty((n_bootstrap, n_stim), dtype=bool)
    M_B = np.empty((n_bootstrap, n_stim), dtype=bool)

    for j, stim in enumerate(stims):
        a = stim2A[stim]; b = stim2B[stim]
        na, nb = len(a), len(b)
    
        idxs_a = rng.integers(0, na, size=(n_bootstrap, na))
        idxs_b = rng.integers(0, nb, size=(n_bootstrap, nb))

        bs_a = a[idxs_a]
        bs_b = b[idxs_b]

        M_A[:, j] = bs_a.sum(axis=1) >= correct_threshold * bs_a.shape[1]
        M_B[:, j] = bs_b.sum(axis=1) >= correct_threshold * bs_b.shape[1]

    # Stim-level bootstrap: sample columns with replacement
    idxs_stim = rng.integers(0, n_stim, size=(n_bootstrap, n_stim))
    rows = np.arange(n_bootstrap)[:, None]

    M_A_bs = M_A[rows, idxs_stim]  # (n_bootstrap, n_stim)
    M_B_bs = M_B[rows, idxs_stim]

    # F1 across bootstraps
    tp = np.sum(M_A_bs & M_B_bs, axis=1)
    fp = np.sum(~M_A_bs & M_B_bs, axis=1)
    fn = np.sum(M_A_bs & ~M_B_bs, axis=1)
    denom = 2*tp + fp + fn

    f1s = np.zeros_like(tp, dtype=float)
    valid = denom > 0
    f1s[valid] = (2 * tp[valid]) / denom[valid]

    # Handle cases where all predictions are correct (denom=0 but tp=total)
    all_correct_mask = (tp == n_stim) & (fp == 0) & (fn == 0)
    f1s[all_correct_mask] = 1.0
    mean_f1 = f1s.mean()
    f1_lo, f1_hi = np.percentile(f1s, [2.5, 97.5])

    # Jaccard for misclassification sets
    misA = ~M_A_bs
    misB = ~M_B_bs

    inter = np.sum(misA & misB, axis=1)
    union = np.sum(misA | misB, axis=1)
    
    valid = union > 0
    jacc_nonzero = inter[valid] / union[valid]

    # Handle empty Jaccard scores
    if len(jacc_nonzero) == 0:
        mean_jacc = 0.0
        jacc_lo, jacc_hi = 0.0, 0.0
    else:
        mean_jacc = jacc_nonzero.mean()
        jacc_lo, jacc_hi = np.percentile(jacc_nonzero, [2.5, 97.5])
    
    n_all_correct = np.sum(all_correct_mask)
    n_no_error    = n_bootstrap - len(jacc_nonzero)
    if n_all_correct > 0:
        print(f"  → All‑correct replicates: {n_all_correct}/{n_bootstrap}")
    if n_no_error > 0:
        print(f"  → No‑error replicates:    {n_no_error}/{n_bootstrap}")
    
    return mean_f1, (f1_lo, f1_hi), mean_jacc, (jacc_lo, jacc_hi), f1s, jacc_nonzero

def bootstrap_consensus_metrics(
    stim2A: Dict[str, np.ndarray],
    stim2B: Dict[str, np.ndarray] = None,
    n_bootstrap: int = 1_000,
    seed: int = None
) -> Tuple[float, Tuple[float, float], float, Tuple[float, float], np.ndarray, np.ndarray]:
    """
    Bootstrap‑based agreement estimator for stimulus‑level responses.

    Overview
    --------
    ▸ **Self‑agreement (bootstrap test–retest)**  
      ‑ Call with only `stim2A`.  
      ‑ For each replicate, draw *two independent bootstrap resamples* of every
        stimulus' trial list; compare the resamples to quantify the model's
        internal consistency.

    ▸ **Cross‑group agreement**  
      ‑ Call with both `stim2A` and `stim2B`.  
      ‑ For each replicate, trials are resampled—with replacement—for **both**
        groups and the resampled stimuli are compared.

    Per‑replicate computation
    -------------------------
    1. Resample trials for every stimulus and group.
    2. Mark a stimulus "correct" for a group if  
       `(# correct trials ≥ correct_threshold · # resampled trials)`.
    3. Over the full stimulus set, compute:  
       • **F1‑score** of the "correct" labels  
       • **Jaccard index** of the error sets (misclassified stimuli).

    Parameters
    ----------
    stim2A, stim2B
        Mapping ``{stimulus_id: np.ndarray[bool]}``, where the Boolean array
        indicates trial‑level correctness (True = correct).
    n_bootstrap
        Number of bootstrap replicates (default 1 000).
    correct_threshold
        Majority‑vote threshold for a stimulus to be counted as correct
        (default 0.5 = simple majority).
    seed
        Seed for the underlying NumPy random generator. ``None`` → nondeterministic.

    Returns
    -------
    mean_f1 : float
        Mean F1‑score across replicates.
    f1_ci   : (float, float)
        95 % percentile confidence interval for the F1‑score.
    mean_jacc : float
        Mean Jaccard index across replicates (error‑set overlap).
    jacc_ci  : (float, float)
        95 % percentile confidence interval for the Jaccard index.
    f1s_raw : np.ndarray
        Raw F1 scores from all bootstrap iterations.
    jacc_raw : np.ndarray
        Raw Jaccard scores from valid bootstrap iterations.

    Notes
    -----
    • A single `np.random.Generator` instance is used throughout the call, so
      every random draw is independent yet reproducible.  
    • Replicates in which both groups are flawless (no errors) are excluded
      from the Jaccard average and reported separately by the caller.
    """
    rng = np.random.default_rng(seed)
    
    # Case 1: Bootstrap from a single DataFrame
    if stim2B is None:
        return _run_bootstrap_analysis(
            stim2A, stim2A,
            n_bootstrap=n_bootstrap,
            rng=rng
        )
    #     f1_scores = []
    #     jacc_scores = []
    #     successful = 0

    #     for _ in tqdm(range(n_bootstrap), desc="Bootstrapping splits"):
    #         # For each stimulus, shuffle the results array
    #         split1 = {}
    #         split2 = {}
            
    #         for stim, results in stim2A.items():
    #             # Shuffle the results for this stimulus
    #             shuffled = rng.permutation(results)
    #             # Split the shuffled array into two halves
    #             mid = len(shuffled) // 2
    #             split1[stim] = shuffled[:mid]
    #             split2[stim] = shuffled[mid:]
            
    #         # Run bootstrap "replicates" 
    #         f1, _, jacc, _ = _run_bootstrap_analysis(
    #             split1, split2,
    #             n_bootstrap=1,              # just one replicate
    #             seed=rng.integers(0, 2**32)
    #         )
    #         f1_scores.append(f1)
    #         jacc_scores.append(jacc)
    #     mean_f1   = np.mean(f1_scores)
    #     f1_lo, f1_hi     = np.percentile(f1_scores, [2.5, 97.5])
    #     mean_jacc = np.mean(jacc_scores)
    #     jacc_lo, jacc_hi = np.percentile(jacc_scores, [2.5, 97.5])
    #     return mean_f1, (f1_lo, f1_hi), mean_jacc, (jacc_lo, jacc_hi)
    # # Case 2: Bootstrap from two dictionaries
    else:
        return _run_bootstrap_analysis(
            stim2A, stim2B, 
            n_bootstrap=n_bootstrap,
            rng=rng
        )

def paired_bootstrap_jaccard(
    stim2_human: Dict[str, np.ndarray],
    stim2_model: Dict[str, np.ndarray],
    n_bootstrap: int = 1_000,
    correct_threshold: float = 0.5,
    rng: np.random.Generator | None = None,
):
    """
    Paired bootstrap test: is human-internal Jaccard > human-model Jaccard?
    Returns (mean_diff, (ci_lo, ci_hi), p_one_tailed, diffs)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Stimuli present in *both* groups and with ≥5 trials each
    stims = [
        s for s in stim2_human
        if s in stim2_model
        and len(stim2_human[s]) >= 5
        and len(stim2_model[s]) >= 5
    ]
    n_stim = len(stims)
    diffs = []

    for _ in range(n_bootstrap):
        H1 = np.empty(n_stim, dtype=bool)
        H2 = np.empty(n_stim, dtype=bool)
        M  = np.empty(n_stim, dtype=bool)

        # ---- bootstrap resampling of trials ----
        for j, stim in enumerate(stims):
            h_trials = stim2_human[stim]
            m_trials = stim2_model[stim]

            n_h, n_m = len(h_trials), len(m_trials)

            # draw WITH replacement
            idx_H1 = rng.integers(0, n_h, n_h)
            idx_H2 = rng.integers(0, n_h, n_h)
            idx_M  = rng.integers(0, n_m, n_m)

            H1[j] = h_trials[idx_H1].sum() >= correct_threshold * n_h
            H2[j] = h_trials[idx_H2].sum() >= correct_threshold * n_h
            M[j]  = m_trials[idx_M].sum()  >= correct_threshold * n_m

        # ---- Jaccard computations ----
        def jaccard(a, b):
            mis_a = ~a
            mis_b = ~b
            union = np.sum(mis_a | mis_b)
            if union == 0:          # both flawless → undefined
                return np.nan
            return np.sum(mis_a & mis_b) / union

        j_int   = jaccard(H1, H2)
        j_cross = jaccard(H1, M)

        if not np.isnan(j_int) and not np.isnan(j_cross):
            diffs.append(j_int - j_cross)

    diffs = np.asarray(diffs)
    mean_diff = diffs.mean()
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_one_tailed = np.mean(diffs <= 0)

    return mean_diff, (ci_lo, ci_hi), p_one_tailed, diffs

def _capitalize_human(arch_name: str) -> str:
    """Capitalize 'human' in architecture names for display purposes."""
    return arch_name.replace('human', 'Human')

if __name__ == "__main__":
    import pickle
    import os
    results_pickle = os.path.join("summary_datasets", "arch_to_stim2results.pkl")
    with open(results_pickle, "rb") as f:
        arch_to_stim2results = pickle.load(f)
   
    # Create matrices to store the comparison results
    architectures = list(arch_to_stim2results.keys())
    n_arch = len(architectures)
    
    # Initialize matrices to store F1 and Jaccard scores
    f1_matrix = np.zeros((n_arch, n_arch))
    jacc_matrix = np.zeros((n_arch, n_arch))
    
    # Initialize lists to store confidence intervals
    f1_ci = {}
    jacc_ci = {}
    
    # Store raw bootstrap arrays for statistical testing
    bootstrap_arrays = {}
    
    # Compute only upper triangle of pairwise comparisons
    print("Computing pairwise comparisons (upper triangle only)...")
    for i, arch1 in enumerate(architectures):
        for j in range(i, n_arch):  # Only compute upper triangle
            arch2 = architectures[j]
            key = (arch1, arch2)
            
            if i == j:  # Self-comparison (split-half reliability)
                m_f1, (f1_lo, f1_hi), m_j, (j_lo, j_hi), f1s, jacc_raw = bootstrap_consensus_metrics(
                    arch_to_stim2results[arch1]
                )
             
            else:  # Cross-architecture comparison
                m_f1, (f1_lo, f1_hi), m_j, (j_lo, j_hi), f1s, jacc_raw = bootstrap_consensus_metrics(
                    arch_to_stim2results[arch1], arch_to_stim2results[arch2]
                )
                
            # Store results in matrices
            f1_matrix[i, j] = m_f1
            jacc_matrix[i, j] = m_j
            
            # Store confidence intervals
            f1_ci[key] = (f1_lo, f1_hi)
            jacc_ci[key] = (j_lo, j_hi)
            
            # Store raw bootstrap arrays
            bootstrap_arrays[key] = {'f1': f1s, 'jaccard': jacc_raw}
    
    # Mirror the upper triangle to fill the lower triangle
    f1_matrix = np.maximum(f1_matrix, f1_matrix.T)
    jacc_matrix = np.maximum(jacc_matrix, jacc_matrix.T)
    
    # Plot matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 6.5))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(f1_matrix), k=1)  # k=1 to exclude diagonal
    
    # Create display labels with capitalized 'human'
    display_labels = [_capitalize_human(arch) for arch in architectures]
    display_labels = [label.replace('Vision Transformer (ViT)', 'ViT') for label in display_labels]
    
    # Define custom color palette
    custom_palette = sns.color_palette("Blues", as_cmap=True)
    
    # Create a figure with a shared colorbar
    fig = plt.figure(figsize=(6.5,3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.05], width_ratios=[1, 1], left=0.2, right=0.9, hspace=0.5)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cbar_ax1 = fig.add_subplot(gs[1, 0])
    cbar_ax2 = fig.add_subplot(gs[1, 1])

    # F1 matrix heatmap
    sns.heatmap(f1_matrix, annot=True, fmt=".2f", cmap="flare", 
                xticklabels=display_labels, yticklabels=display_labels, ax=ax1,
                mask=mask, cbar_ax=cbar_ax1, cbar_kws={'orientation': 'horizontal'})
    ax1.set_title("F1 Score")
    plt.setp(ax1.get_xticklabels(), rotation=20, ha='right')
    
    # Jaccard matrix heatmap
    sns.heatmap(jacc_matrix, annot=True, fmt=".2f", cmap="mako", 
                xticklabels=display_labels, yticklabels=display_labels, ax=ax2,
                mask=mask, cbar_ax=cbar_ax2, cbar_kws={'orientation': 'horizontal'})
    ax2.set_title("Jaccard Index")
    plt.setp(ax2.get_xticklabels(), rotation=20, ha='right')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(Path("plots/model_comparison_matrix.pdf"))
    plt.close()

    
    # Statistical testing: Compare human internal alignment vs best architecture alignment
    print("\n" + "="*60)
    print("STATISTICAL TESTING")
    print("="*60)
    
    # Find human architecture
    human_idx = None
    for i, arch in enumerate(architectures):
        if 'human' in arch.lower():
            human_idx = i
            break
    
    if human_idx is not None:
        human_arch = architectures[human_idx]
        print(f"Human architecture found: {human_arch}")
        
        # Get human internal alignment (self-comparison)
        human_key = (human_arch, human_arch)
        human_internal_jacc = bootstrap_arrays[human_key]['jaccard']
        print(f"Human internal alignment (Jaccard): {jacc_matrix[human_idx, human_idx]:.3f} "
              f"(n={len(human_internal_jacc)} valid bootstraps)")
        
        # Find the architecture with highest alignment to humans
        human_row = jacc_matrix[human_idx, :]
        # Exclude self-comparison
        human_row_excl_self = human_row.copy()
        human_row_excl_self[human_idx] = -1  # Set self-comparison to -1 to exclude it
        
        best_arch_idx = np.argmax(human_row_excl_self)
        best_arch = architectures[best_arch_idx]
        best_alignment = human_row[best_arch_idx]
        
        print(f"Architecture with highest human alignment: {best_arch}")
        print(f"Human-{best_arch} alignment (Jaccard): {best_alignment:.3f}")
        
        # Get the bootstrap arrays for comparison
        best_arch_key = (human_arch, best_arch) if human_idx < best_arch_idx else (best_arch, human_arch)
        best_arch_jacc = bootstrap_arrays[best_arch_key]['jaccard']
        print(f"Human-{best_arch} alignment (n={len(best_arch_jacc)} valid bootstraps)")
        
        mean_d, (lo, hi), p, diffs = paired_bootstrap_jaccard(
                stim2_human   = arch_to_stim2results['human'],
                stim2_model   = arch_to_stim2results[best_arch],
                rng           = np.random.default_rng(42),
        )

        print(f"Mean Δ(Jaccard): {mean_d:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        print(f"One-tailed p-value: {p:.6f}")
        
    else:
        print("No human architecture found in the dataset!")
    
    print("="*60)

    # Create list of tuples (arch_pair, f1, jaccard) for sorting
    results = []
    for i, arch1 in enumerate(architectures):
        for j in range(i, n_arch):
            arch2 = architectures[j]
            if i == j:
                pair = f"{arch1} (split-half)"
            else:
                pair = f"{arch1} vs {arch2}"
            key = (arch1, arch2)
            results.append((pair, f1_matrix[i,j], jacc_matrix[i,j], f1_ci[key], jacc_ci[key]))
    
    # Sort by Jaccard index in descending order
    results.sort(key=lambda x: x[2], reverse=True)
    # Convert results to dictionary format for JSON serialization
    results_dict = {
        'comparisons': [
            {
                'pair': pair,
                'f1_score': float(f1),
                'jaccard_index': float(jacc),
                'f1_ci': (float(f1_ci[0]), float(f1_ci[1])),
                'jaccard_ci': (float(jacc_ci[0]), float(jacc_ci[1]))
            }
            for pair, f1, jacc, f1_ci, jacc_ci in results
        ],
        'bootstrap_arrays': {
            f"{key[0]}_vs_{key[1]}": {
                'f1_scores': bootstrap_arrays[key]['f1'].tolist(),
                'jaccard_scores': bootstrap_arrays[key]['jaccard'].tolist()
            }
            for key in bootstrap_arrays.keys()
        }
    }
    
    # Save results to JSON file
    output_path = Path('plots/trial_level.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
        
    # Save matrices to numpy files
    np.save(Path('plots/f1_matrix.npy'), f1_matrix)
    np.save(Path('plots/jaccard_matrix.npy'), jacc_matrix)
    
    # Save architecture names for reference
    with open(Path('plots/architectures.txt'), 'w') as f:
        f.write('\n'.join(architectures))