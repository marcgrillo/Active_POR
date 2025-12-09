import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import wilcoxon
import common.utils as utils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_metric_results(metric_name, F1, F2, F3, hm, num_dm_dec, dataset_fold=None, sub_fold=None, drop_index=None, save_figs=False, fig_name_prefix="figure"):
    """
    Generic plotting function for ASRS, ASPS, or AIOS results.
    Plots Mean +/- 95% Confidence Interval and the Ratio of Active/Regular.
    """
    metric_name = metric_name.lower()
    valid_metrics = ['asrs', 'asps', 'aios', 'perc_inc']
    if metric_name not in valid_metrics:
        raise ValueError(f"metric_name must be one of {valid_metrics}")

    # Configuration
    f1, f2, f3 = F1[0], F2[0], F3[0]
    x = np.arange(1, num_dm_dec + 1)

    # Load Data
    y, y_active = utils.load_test_results(metric_name, dataset_fold, sub_fold, num_dm_dec, f1, f2, f3)
    
    if y.size == 0 or y_active.size == 0:
        print(f"No data found for {metric_name} in {dataset_fold}/{sub_fold}")
        return

    # Drop outliers if requested
    if drop_index is not None:
        y = np.delete(y, drop_index, axis=1)
        y_active = np.delete(y_active, drop_index, axis=1)

    # Statistics
    n_samples = y.shape[1] 
    mean = np.mean(y, axis=1)
    std = np.std(y, axis=1) * 2 / np.sqrt(n_samples) # 95% CI
    
    mean_active = np.mean(y_active, axis=1)
    std_active = np.std(y_active, axis=1) * 2 / np.sqrt(n_samples) # 95% CI

    save_dir = os.path.join("figs", metric_name)
    if save_figs:
        os.makedirs(save_dir, exist_ok=True)

    # --- Plot 1: Evolution ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean, label="Regular", color="blue")
    plt.fill_between(x, mean - std, mean + std, color="blue", alpha=0.3, label="95% CI Regular")
    plt.plot(x, mean_active, label="Active", color="orange")
    plt.fill_between(x, mean_active - std_active, mean_active + std_active, color="orange", alpha=0.3, label="95% CI Active")

    plt.xlabel("Number of DM Preferences", fontsize=12)
    plt.ylabel(metric_name.upper(), fontsize=12)
    plt.title(f"{metric_name.upper()} Evolution (N={n_samples})", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_figs:
        plt.savefig(os.path.join(save_dir, f"{fig_name_prefix}_mean_std.png"), dpi=300)
    plt.show()

    # --- Plot 2: Ratio ---
    ratio = np.divide(mean_active, mean, out=np.ones_like(mean_active), where=mean != 0)
    
    """plt.figure(figsize=(10, 6))
    plt.plot(x, ratio, label="Active / Regular", color="purple")
    plt.axhline(y=1, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Number of DM Preferences", fontsize=12)
    plt.ylabel("Ratio", fontsize=12)
    plt.title(f"{metric_name.upper()} Ratio (Active / Regular)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_figs:
        plt.savefig(os.path.join(save_dir, f"{fig_name_prefix}_ratio.png"), dpi=300)
    plt.show()"""

def plot_wilcoxon_test(metric_name, F1, F2, F3, hm, num_dm_dec, dataset_fold=None, sub_folds=None, drop_index=None, alternative = 'greater', save_figs=False, fig_name_prefix="figure"):
    """
    Performs the Wilcoxon signed-rank test between Active and Regular strategies 
    at each step and plots the p-values for multiple sub_folds.
    
    Args:
        sub_folds (list or str): A single sub_fold string or a list of sub_fold strings.
    """
    metric_name = metric_name.lower()
    valid_metrics = ['asrs', 'asps', 'aios', 'perc_inc']
    if metric_name not in valid_metrics:
        raise ValueError(f"metric_name must be one of {valid_metrics}")

    if isinstance(sub_folds, str):
        sub_folds = [sub_folds]

    f1, f2, f3 = F1[0], F2[0], F3[0]
    x = np.arange(1, num_dm_dec + 1)

    plt.figure(figsize=(12, 7))
    
    # Iterate over each sub_fold (method)
    for sub_fold in sub_folds:
        # Load Data
        y, y_active = utils.load_test_results(metric_name, dataset_fold, sub_fold, num_dm_dec, f1, f2, f3)
        
        if y.size == 0 or y_active.size == 0:
            print(f"No data found for {sub_fold}. Skipping.")
            continue

        if drop_index is not None:
            y = np.delete(y, drop_index, axis=1)
            y_active = np.delete(y_active, drop_index, axis=1)

        p_values = []
        
        # Calculate Wilcoxon for each step
        for i in range(num_dm_dec):
            reg_scores = y[i, :]
            act_scores = y_active[i, :]
            
            if np.allclose(reg_scores, act_scores):
                p_values.append(1.0)
            else:
                try:
                    res = wilcoxon(act_scores, y = reg_scores, alternative = alternative)
                    p_values.append(res.pvalue)
                except ValueError:
                    p_values.append(1.0)

        # Plot p-values for this sub_fold
        plt.plot(x, p_values, label=f"{sub_fold}", linewidth=1.5)

    # Plot Settings
    plt.axhline(y=0.05, color="red", linestyle="--", linewidth=1.5, label="p=0.05 (Significance)")
    plt.axhline(y=0.01, color="darkred", linestyle=":", linewidth=1.5, label="p=0.01")

    plt.yscale("log")
    plt.xlabel("Number of DM Preferences", fontsize=12)
    plt.ylabel("p-value (Log Scale)", fontsize=12)
    plt.title(f"Wilcoxon Signed-Rank Test: {metric_name.upper()} (Active vs Regular)", fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_dir = os.path.join("figs", metric_name)
    if save_figs:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{fig_name_prefix}_wilcoxon_multi.png"), dpi=300)
    
    plt.show()

def plot_var_cand(x, y, labels=None, save_path=None):
    """
    Plots mean and spread. 
    Highlights max X (Red) and max Y (Green) and forces them to the top layer.
    """
    
    # --- 1. Data Processing ---
    X_mat = np.array(x)
    Y_mat = np.array(y)
    
    if X_mat.shape != Y_mat.shape:
        raise ValueError(f"Shape mismatch: x is {X_mat.shape}, y is {Y_mat.shape}")

    # Calculate statistics
    x_mean = np.nanmean(X_mat, axis=0)
    y_mean = np.nanmean(Y_mat, axis=0)
    x_err = np.nanstd(X_mat, axis=0)
    y_err = np.nanstd(Y_mat, axis=0)
    
    # --- 2. Identify Indices ---
    idx_max_x = np.argmax(x_mean)
    idx_max_y = np.argmax(y_mean)
    
    # Create a list of indices excluding the special ones
    all_indices = np.arange(len(x_mean))
    mask_normal = (all_indices != idx_max_x) & (all_indices != idx_max_y)
    
    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # A) Plot Error Bars for EVERYONE (Background layer)
    # zorder=1: Lowest layer
    ax.errorbar(x_mean, y_mean, xerr=x_err, yerr=y_err, 
                fmt='none', ecolor='gray', alpha=0.5, capsize=3, zorder=1)

    # B) Plot NORMAL points (Middle layer)
    # zorder=2: Sits on top of lines, but below highlights
    ax.scatter(x_mean[mask_normal], y_mean[mask_normal], 
               s=60, c='tab:blue', edgecolors='black', label='Candidates', zorder=2)

    # C) Plot SPECIAL points (Top layer)
    # zorder=3: Sits on top of everything
    # 1. Max X (Red)
    ax.scatter(x_mean[idx_max_x], y_mean[idx_max_x], 
               s=100, c='red', edgecolors='black', zorder=3)
    
    # 2. Max Y (Green)
    # Note: If a point is BOTH max X and max Y, it will appear Green (drawn last)
    ax.scatter(x_mean[idx_max_y], y_mean[idx_max_y], 
               s=100, c='green', edgecolors='black', zorder=3)

    # Annotations
    if labels:
        for i, txt in enumerate(labels):
            # Check if this point is a highlighted one to give it bold text or different offset
            is_highlight = (i == idx_max_x) or (i == idx_max_y)
            font_weight = 'bold' if is_highlight else 'normal'
            
            ax.annotate(txt, (x_mean[i], y_mean[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, weight=font_weight)

    # Custom Legend
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markeredgecolor='k', label='Candidates (Mean ± Std)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='k', label=f'Max H (idx {idx_max_x})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markeredgecolor='k', label=f'Max MI (idx {idx_max_y})'),
    ]
    ax.legend(handles=legend_handles)

    ax.set_title(f'Candidate Performance: Maxima Highlights ({X_mat.shape[0]} Reps)')
    ax.set_xlabel('H')
    ax.set_ylabel('MI')
    ax.grid(True, linestyle='--', alpha=0.5)

    # --- 4. Smart Save Logic ---
    if save_path:
        filename, extension = os.path.splitext(save_path)
        counter = 1
        
        final_path = save_path
        while os.path.exists(final_path):
            final_path = f"{filename}_{counter}{extension}"
            counter += 1
            
        plt.savefig(final_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {final_path}")
    
    plt.close()