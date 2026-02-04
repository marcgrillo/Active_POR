import os
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import nnls
from tqdm import tqdm
from common import utils
from mcda.models import PiecewiseLinearTransformer

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATASET_FOLD = 'datasets'
F1_LIST = [30]
F2_LIST = [4]
F3_LIST = [100]

TARGET_METHODS = [
    'FTRL_BT_BALD',
    # Add other methods here if needed
]

OUTPUT_DIR = 'plots_analysis'
utils.save_path(OUTPUT_DIR)

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------

def get_best_fit_pwl(table, true_u, num_intervals=3):
    """
    Fits the best Piecewise Linear (PWL) parameters w* to the True Utility U
    using Non-Negative Least Squares (NNLS).
    """
    transformer = PiecewiseLinearTransformer.from_equal_intervals(table, num_intervals)
    features = transformer.transform(table)
    
    # Solve: min || Xw - U ||_2 subject to w >= 0
    w_star, residual = nnls(features, true_u)
    
    # Normalize to sum to 1 (since our models usually assume simplex)
    # Note: If true U scale is arbitrary, this normalization maps it to our simplex space.
    sum_w = np.sum(w_star)
    if sum_w > 1e-9:
        w_star = w_star / sum_w
    else:
        # Fallback if w is all zeros (shouldn't happen with positive utility)
        w_star = np.ones_like(w_star) / len(w_star)
        
    return w_star, transformer

def evaluate_pwl(transformer, w, f2, num_intervals=3, resolution=100):
    """
    Evaluates the marginal utility functions for plotting.
    Returns: dict {criterion_idx: (x_vals, y_vals)}
    """
    curves = {}
    current_idx = 0
    
    # Global min/max for the whole table usually used for breakpoints
    # But transformer stores specific breakpoints per criterion
    
    for j in range(f2):
        breakpoints = transformer.ch_p[j]
        n_w = len(breakpoints) - 1
        
        w_sub = w[current_idx : current_idx + n_w]
        
        x_vals = np.linspace(breakpoints[0], breakpoints[-1], resolution)
        y_vals = []
        
        for x in x_vals:
            val = 0
            for k in range(n_w):
                bp_low = breakpoints[k]
                bp_high = breakpoints[k+1]
                # Re-implement transform logic for single scalar
                feat = np.clip((x - bp_low) / (bp_high - bp_low), 0, 1)
                val += w_sub[k] * feat
            y_vals.append(val)
            
        curves[j] = (x_vals, np.array(y_vals))
        current_idx += n_w
        
    return curves

def plot_marginal_comparison(transformer, w_star, w_inferred, f2, save_path, title_suffix="", true_w=None, true_params=None, utility_type='exponential'):
    """
    Plots Best-Fit vs Inferred Marginal Utilities.
    If true_w and true_params are provided, plots the True function.
    """
    curves_star = evaluate_pwl(transformer, w_star, f2)
    curves_inf = evaluate_pwl(transformer, w_inferred, f2)
    
    fig, axes = plt.subplots(1, f2, figsize=(5 * f2, 4))
    if f2 == 1: axes = [axes]
    
    for j in range(f2):
        ax = axes[j]
        
        # Plot Best Fit PWL (Approximation)
        x_star, y_star = curves_star[j]
        ax.plot(x_star, y_star, 'k--', linewidth=2, label='Best Fit PWL', alpha=0.7)
        
        # Plot Inferred PWL
        x_inf, y_inf = curves_inf[j]
        ax.plot(x_inf, y_inf, 'r-', linewidth=2, label='Inferred PWL')
        
        # Plot True Analytical (if available and exponential)
        if utility_type == 'exponential' and true_w is not None and true_params is not None:
             w_j = true_w[j]
             c_j = true_params[j]
             
             # Assuming domain 0-1 for plotting logic (since features are normalized usually?)
             # Wait, transformer.ch_p has real values.
             # We should plot over the range [min, max] of the criterion.
             min_val = x_star[0]
             max_val = x_star[-1]
             x_plot = np.linspace(min_val, max_val, 100)
             
             # Formula: y = w_j * (1 - exp(-c * x)) / (1 - exp(-c))
             # BUT: The dataset generation used `1 - exp(-c_j * data_matrix)`. 
             # Does it assume data is 0-1? 
             # `generate_datasets.py` uses `ls = np.random.rand(f1, f2)`. Yes, data is 0-1.
             
             # So min_val approx 0, max_val approx 1.
             term_denom = 1 - np.exp(-c_j)
             y_true = w_j * (1 - np.exp(-c_j * x_plot)) / term_denom
             
             ax.plot(x_plot, y_true, 'g:', linewidth=2.5, label='True Exponential')

        ax.set_title(f"Criterion {j+1}")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend()
            
    plt.suptitle(f"Marginal Utility Comparison {title_suffix}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    for f1 in F1_LIST:
        for f2 in F2_LIST:
            for f3 in F3_LIST:
                print(f"\nProcessing Config: F1={f1}, F2={f2}, F3={f3}")
                
                # 1. Load Dataset
                try:
                    tables, rankings, coupless, Us = utils.read_dataset(DATASET_FOLD, f1, f2, f3)
                except FileNotFoundError:
                    print("Dataset not found, skipping.")
                    continue
                
                if Us is None:
                    print("No True Utilities found (Us.csv), cannot compute distance.")
                    continue

                # Load Generation Params to know utility type
                gen_params_path = os.path.join(DATASET_FOLD, "generation_params.json")
                utility_type = 'exponential' # Default
                if os.path.exists(gen_params_path):
                    with open(gen_params_path, 'r') as f:
                        gp = json.load(f)
                        utility_type = gp.get("utility_type", "exponential")
                
                # Load True Params (c) if exponential
                true_params = None
                if utility_type == 'exponential':
                    base_name = f"f1_{f1}__f2_{f2}__ndm" # Partial match
                    # Find the full filename roughly or construct it
                    # The script saves as base_name + "params.csv"
                    # We need to know num_dm_dec to construct filename exactly, which we calculate below
                    pass # Will load after calculating num_dm_dec
                
                num_dm_dec = int(np.round(f3 * (f1 * (f1 - 1) / 200)))
                
                # Load weights and params
                try:
                    base_name = f"f1_{f1}__f2_{f2}__ndm_{num_dm_dec}"
                    w_path = os.path.join(DATASET_FOLD, f"{base_name}weights.csv")
                    true_weights_all = np.loadtxt(w_path)
                    
                    true_params_all = None
                    if utility_type == 'exponential':
                        p_path = os.path.join(DATASET_FOLD, f"{base_name}params.csv")
                        if os.path.exists(p_path):
                            true_params_all = np.loadtxt(p_path)
                except Exception as e:
                    print(f"Error loading extra params: {e}")
                    true_weights_all = None
                
                # 2. Iterate Methods
                for sub_fold in TARGET_METHODS:
                    print(f"  Method: {sub_fold}")
                    alg_name, active_method = utils.parse_subfold_string(sub_fold)
                    
                    method_dir = os.path.join(f"samples_{DATASET_FOLD}", f"f1_{f1}_f2_{f2}_f3_{f3}", sub_fold)
                    if not os.path.exists(method_dir):
                        print(f"    Results directory not found: {method_dir}")
                        continue
                        
                    all_distances = [] # Shape: (HM, num_steps)
                    
                    # 3. Process Each Human Model
                    hm_limit = len(tables)
                    
                    for i in tqdm(range(hm_limit), desc="    Human Models"):
                        table = tables[i]
                        true_u = Us[i]
                        
                        # A. Get Ground Truth Representative (w*)
                        w_star, transformer = get_best_fit_pwl(table, true_u, num_intervals=3)
                        
                        # B. Load Inferred Trajectory
                        table_dir = os.path.join(method_dir, f"table_{i}")
                        distances_hm = []
                        w_final_inferred = None
                        
                        for step in range(1, num_dm_dec + 1):
                            path_active = os.path.join(table_dir, f"{step}_active.npy")
                            
                            if os.path.exists(path_active):
                                w_inf = np.load(path_active)
                                
                                # L2 Distance
                                dist = np.linalg.norm(w_star - w_inf)
                                distances_hm.append(dist)
                                
                                w_final_inferred = w_inf
                            else:
                                distances_hm.append(np.nan)
                        
                        all_distances.append(distances_hm)
                        
                        # C. Qualitative Plot (Only for first HM)
                        if i == 0 and w_final_inferred is not None:
                            plot_name = f"marginals_{sub_fold}_f1{f1}f2{f2}_HM{i}.png"
                            
                            # Get true w and params for this HM
                            t_w = true_weights_all[i] if true_weights_all is not None else None
                            t_p = true_params_all[i] if true_params_all is not None else None
                            
                            plot_marginal_comparison(
                                transformer, 
                                w_star, 
                                w_final_inferred, 
                                f2, 
                                os.path.join(OUTPUT_DIR, plot_name),
                                title_suffix=f"({alg_name}, HM {i}, Final Step)",
                                true_w=t_w,
                                true_params=t_p,
                                utility_type=utility_type
                            )

                    # 4. Aggregate & Plot Distances
                    all_distances = np.array(all_distances) # (HM, Steps)
                    
                    # Handle NaNs if runs crashed or incomplete
                    valid_mask = ~np.isnan(all_distances).any(axis=1)
                    if np.sum(valid_mask) == 0:
                        print("    No complete runs found.")
                        continue
                        
                    valid_distances = all_distances[valid_mask]
                    mean_dist = np.mean(valid_distances, axis=0)
                    std_dist = np.std(valid_distances, axis=0) / np.sqrt(len(valid_distances)) # Standard Error
                    
                    # Plot
                    steps = np.arange(1, len(mean_dist) + 1)
                    plt.figure(figsize=(8, 6))
                    plt.plot(steps, mean_dist, label=f"{alg_name} (L2)", color='b')
                    plt.fill_between(steps, mean_dist - std_dist, mean_dist + std_dist, color='b', alpha=0.2)
                    
                    plt.title(f"Parameter L2 Distance vs DM Preferences\n(F1={f1}, F2={f2})")
                    plt.xlabel("Number of DM Preferences")
                    plt.ylabel("L2 Distance ||w* - w_inf||")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    out_name = f"dist_L2_{sub_fold}_f1{f1}f2{f2}.png"
                    plt.savefig(os.path.join(OUTPUT_DIR, out_name))
                    plt.close()
                    print(f"    Saved distance plot to {out_name}")

