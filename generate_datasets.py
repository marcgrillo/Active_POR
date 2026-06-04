import os
import json
import numpy as np
from scipy.stats import dirichlet
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
from common import utils
from mcda.models import PiecewiseLinearTransformer

class DatasetGenerator:
    """
    Generates synthetic MCDA datasets with controllable Inconsistency.
    Saves generation parameters (lambda) for reproducibility.
    """

    def __init__(self, output_dir="datasets", n_runs=200):
        self.output_dir = output_dir
        self.n_runs = n_runs
        utils.save_path(self.output_dir)

    def _compute_rank(self, U):
        """Computes ranks (0 = best)."""
        b = np.argsort(-U)
        ranks = np.empty_like(b)
        ranks[b] = np.arange(len(U))
        return ranks

    def _generate_exponential_utility(self, data_matrix, f2):
        """Generates ground truth utility using Exponential function."""
        w = dirichlet.rvs(alpha=np.ones(f2))[0]
        c = np.random.uniform(-10, 10, f2)
        c[np.abs(c) < 1e-5] = 1e-5 # Avoid c=0
            
        term1 = 1 - np.exp(-c * data_matrix)
        term2 = 1 - np.exp(-c)
        u_j = w * (term1 / term2)
        return np.sum(u_j, axis=1), w, c

    def _generate_piecewise_linear_utility(self, data_matrix, f2, num_intervals=3):
        """Generates ground truth utility using Piecewise Linear functions."""
        transformer = PiecewiseLinearTransformer.from_equal_intervals(data_matrix, num_intervals)
        features = transformer.transform(data_matrix)
        
        # Features dim: (N, total_params). total_params = f2 * num_intervals
        total_params = features.shape[1]
        
        # Weights
        w = dirichlet.rvs(alpha=np.ones(total_params))[0]
        
        # Utility U = X_transformed @ w
        U = features @ w
        return U, w, transformer

    def _plot_marginal_utilities(self, utility_type, f2, weights, params, save_path):
        """Plots the marginal utility functions for each criterion."""
        x_plot = np.linspace(0, 1, 100)
        
        fig, axes = plt.subplots(1, f2, figsize=(5 * f2, 4))
        if f2 == 1: axes = [axes]
        
        for j in range(f2):
            w_j = weights[j] if utility_type == 'exponential' else 1.0 # Weights are handled differently
            
            if utility_type == 'exponential':
                # exponential: u_j(x) = w_j * (1 - exp(-c_j * x)) / (1 - exp(-c_j))
                c_j = params[j]
                y_plot = w_j * (1 - np.exp(-c_j * x_plot)) / (1 - np.exp(-c_j))
                title = f"Crit {j+1} (Exp, w={w_j:.2f}, c={c_j:.2f})"
            elif utility_type == 'piecewise_linear':
                # piecewise: params is transformer
                transformer = params
                
                # Reconstruct piecewise function
                # Determine relevant weights subset for criterion j
                col_end = np.cumsum(transformer.n_w)
                col_start = np.insert(col_end[:-1], 0, 0)
                
                start_idx = int(col_start[j])
                end_idx = int(col_end[j])
                
                w_sub = weights[start_idx:end_idx]
                breakpoints = transformer.ch_p[j] # Should be [min, ..., max]
                
                # We assume data was normalized 0-1 for plotting if not implicit
                # But transformer uses raw data range. 
                # For visualization, let's plot over the actual domain [min, max]
                # x_plot needs to be scaled to domain or we just plot characteristic points.
                
                # Let's plot the piecewise segments directly
                # Points: (bp_0, 0), (bp_1, u(bp_1)), ...
                # But wait, the formulation is U = sum w_k * h_k(x)
                # h_k(x) is 1 if x > bp_k+1, linear in [bp_k, bp_k+1], 0 else?
                # Actually PiecewiseLinearTransformer implementation transforms x into
                # local coordinates for each interval.
                # U(x) = sum_{k} w_{jk} * phi_{jk}(x)
                
                # To plot: evaluate U_j(x) for x in range
                # Or just evaluate at breakpoints.
                
                # Since w are random dirichlet, they sum to 1 across ALL criteria/segments.
                # The total importance of criterion j is sum(w_sub).
                
                # Evaluation at breakpoints:
                # Value starts at 0 (implicitly)
                # Accumulate value?
                # Let's look at transform:
                # features[:, current_col_idx + k] = np.clip(val, 0, 1)
                # So for x > bp_high, feature is 1.
                # U(x) = sum(w_k * clip((x-low)/(high-low)))
                
                x_domain = np.linspace(breakpoints[0], breakpoints[-1], 200)
                y_vals = []
                for x_val in x_domain:
                    val_j = 0
                    for k in range(len(w_sub)):
                        bp_low = breakpoints[k]
                        bp_high = breakpoints[k+1]
                        feat = np.clip((x_val - bp_low) / (bp_high - bp_low), 0, 1)
                        val_j += w_sub[k] * feat
                    y_vals.append(val_j)
                
                x_plot = x_domain
                y_plot = np.array(y_vals)
                title = f"Crit {j+1} (PWL, imp={np.sum(w_sub):.2f})"

            ax = axes[j]
            ax.plot(x_plot, y_plot, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Attribute Value")
            ax.set_ylabel("Marginal Utility")
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _calculate_expected_inconsistency(self, lam, utility_diffs, model='logit'):
        """P(flip) = 1 - P(correct)."""
        if model == 'logit':
            # P(correct) = sigmoid(lam * delta_U)
            probs = 1.0 - utils.robust_sigmoid(lam * utility_diffs)
        elif model == 'linear':
             # P(correct) = 0.5 * (1 + delta_U). No lambda used (or lambda=1)
             # Note: lambda is ignored here as per requirements
             p_correct = 0.5 * (1.0 + utility_diffs)
             p_correct = np.clip(p_correct, 0.0, 1.0)
             probs = 1.0 - p_correct
        else:
            raise ValueError(f"Unknown model: {model}")
            
        return np.mean(probs)

    def _tune_lambda(self, f1, f2, target_inconsistency, model='logit', num_intervals=3, utility_type='exponential'):
        """Finds the lambda value that results in the desired inconsistency rate."""
        if model == 'linear':
            return 1.0 # Lambda irrelevant for linear model

        if target_inconsistency <= 0: return 1000.0
        if target_inconsistency >= 0.5: return 0.0

        # 1. Calibration set of Utility Differences
        calibration_size = 1000
        diffs = []
        for _ in range(calibration_size):
            ls = np.random.rand(f1, f2)
            if utility_type == 'piecewise_linear':
                U, _, _ = self._generate_piecewise_linear_utility(ls, f2, num_intervals)
            else:
                U, _, _ = self._generate_exponential_utility(ls, f2)
            idx = np.arange(f1)
            np.random.shuffle(idx)
            u_shuffled = U[idx]
            d = np.abs(u_shuffled[:-1] - u_shuffled[1:])
            diffs.append(d)
        
        utility_diffs = np.concatenate(diffs)

        # 2. Root finding
        def objective(lam):
            return self._calculate_expected_inconsistency(lam, utility_diffs, model=model) - target_inconsistency

        try:
            y_min = objective(0.01)
            y_max = objective(100.0)
            if y_min * y_max > 0:
                return 0.01 if abs(y_min) < abs(y_max) else 100.0
            return brentq(objective, 0.01, 100.0)
        except ValueError:
            return 10.0

    def generate_batch(self, F1, F2, F3, target_inconsistency=0.0, utility_type='exponential', prob_model='logit', num_intervals=3, plot_utilities=True):
        print(f"Generating datasets in '{self.output_dir}' with Inconsistency={target_inconsistency} (Model={prob_model}, Utility={utility_type})...")
        if prob_model == 'linear':
            print("Warning: target_inconsistency ignored for linear probability model.")
        
        # Registry to store lambda values for benchmarking
        params_registry = {
            "target_inconsistency": target_inconsistency,
            "utility_type": utility_type,
            "prob_model": prob_model,
            "num_intervals": num_intervals
        }
        
        for f1 in tqdm(F1, desc="Alternatives (F1)"):
            all_couples = utils.get_combinations(np.arange(f1))
            indices = np.arange(len(all_couples))
            
            for f2 in F2:
                # --- Step 1: Auto-tune Lambda ---
                if target_inconsistency > 0 and prob_model != 'linear':
                    lam = self._tune_lambda(f1, f2, target_inconsistency, model=prob_model, num_intervals=num_intervals, utility_type=utility_type)
                elif prob_model == 'linear':
                    lam = 1.0 # Irrelevant
                else:
                    lam = int(1e9)  # Effectively no inconsistency
                
                # Save lambda for this configuration
                config_key = f"f1_{f1}_f2_{f2}"
                params_registry[config_key] = lam

                for f3 in F3:
                    num_dm_dec = int(np.round(f3 * len(all_couples) / 100))
                    
                    num_dm_dec = int(np.round(f3 * len(all_couples) / 100))
                    
                    lss_batch, ranks_prefs_batch, Us_batch, Weights_batch, Params_batch = [], [], [], [], []
                    
                    for i in range(self.n_runs):
                        ls = np.random.rand(f1, f2)
                        if utility_type == 'piecewise_linear':
                            U, w, params = self._generate_piecewise_linear_utility(ls, f2, num_intervals)
                        else:
                            U, w, params = self._generate_exponential_utility(ls, f2)
                            Params_batch.append(params)
                        
                        rk = self._compute_rank(U)
                        
                        base_name = f"f1_{f1}__f2_{f2}__ndm_{num_dm_dec}"
                        
                        # Plot Utilities (only for the first run to avoid spam)
                        if plot_utilities and i == 0:
                            plot_path = os.path.join(self.output_dir, f"{base_name}_utils_table0.png")
                            self._plot_marginal_utilities(utility_type, f2, w, params, plot_path)

                        
                        np.random.shuffle(indices)
                        selected_pairs = all_couples[indices[:num_dm_dec]]
                        
                        sorted_couples = []
                        for pair in selected_pairs:
                            if U[pair[0]] > U[pair[1]]:
                                winner, loser = pair[0], pair[1]
                            else:
                                winner, loser = pair[1], pair[0]
                            
                            # Inconsistency Injection
                            u_diff = U[winner] - U[loser]
                            
                            if prob_model == 'logit':
                                threshold = utils.robust_sigmoid(lam * u_diff)
                            elif prob_model == 'linear':
                                threshold = 0.5 * (1.0 + u_diff)
                                threshold = np.clip(threshold, 0.0, 1.0)
                            else:
                                raise ValueError(f"Unknown prob_model: {prob_model}")
                            
                            if np.random.rand() > threshold:
                                sorted_couples.append([loser, winner]) # Swap
                            else:
                                sorted_couples.append([winner, loser]) # Correct
                            
                        lss_batch.append(ls)
                        Us_batch.append(U)
                        Weights_batch.append(w)
                        flat_pairs = np.hstack(sorted_couples)
                        row_data = np.hstack([rk, flat_pairs])
                        ranks_prefs_batch.append(row_data)
                    
                    base_name = f"f1_{f1}__f2_{f2}__ndm_{num_dm_dec}"
                    np.savetxt(os.path.join(self.output_dir, f"{base_name}table.csv"), np.vstack(lss_batch))
                    np.savetxt(os.path.join(self.output_dir, f"{base_name}rank+preferences.csv"), np.vstack(ranks_prefs_batch))
                    np.savetxt(os.path.join(self.output_dir, f"{base_name}Us.csv"), np.vstack(Us_batch))
                    np.savetxt(os.path.join(self.output_dir, f"{base_name}weights.csv"), np.vstack(Weights_batch))
                    
                    if utility_type == 'exponential' and Params_batch:
                         np.savetxt(os.path.join(self.output_dir, f"{base_name}params.csv"), np.vstack(Params_batch))

        # --- Save Parameters to JSON ---
        params_file = os.path.join(self.output_dir, "generation_params.json")
        with open(params_file, "w") as f:
            json.dump(params_registry, f, indent=4)
        print(f"Generation parameters saved to {params_file}")

    def verify_consistency(self, F1, F2, F3):
        """Checks actual consistency of generated files."""
        print("\nVerifying dataset consistency...")
        good, total = 0, 0
        
        for f1 in F1:
            for f2 in F2:
                for f3 in F3:
                    try:
                        _, rankings, prefs, _ = utils.read_dataset(self.output_dir, f1, f2, f3)
                        for i in range(len(rankings)):
                            current_rank = rankings[i]
                            for pair in prefs[i]:
                                w, l = int(pair[0]), int(pair[1])
                                if current_rank[w] < current_rank[l]:
                                    good += 1
                                total += 1
                    except Exception as e:
                        print(f"Skipping {f1}/{f2}: {e}")

        consistency = (good / total * 100) if total > 0 else 0
        print(f"Measured Consistency: {consistency:.2f}% (Target Inconsistency: {100-consistency:.2f}%)")
        return consistency

if __name__ == "__main__":
    # --- Configuration ---
    DATASET_NAME = "30_dataset"  # Change this to group datasets (e.g., 'exp1', 'inconsistency_test')
    OUTPUT_DIR = os.path.join("datasets", DATASET_NAME)
    N_RUNS = 200
    
    F1_LIST = [30]
    F2_LIST = [4]
    F3_LIST = [100]
    
    TARGET_INCONSISTENCY = 0.3
    UTILITY_TYPE = 'exponential'  # Options: 'exponential', 'piecewise_linear'
    PROB_MODEL = 'logit'          # Options: 'logit', 'linear'
    NUM_INTERVALS = 3             # Only for piecewise_linear
    PLOT_UTILITIES = True
    
    # --- Execution ---
    print(f"Starting Generation: {UTILITY_TYPE} + {PROB_MODEL}")
    gen = DatasetGenerator(output_dir=OUTPUT_DIR, n_runs=N_RUNS)
    
    gen.generate_batch(
        F1=F1_LIST, 
        F2=F2_LIST, 
        F3=F3_LIST, 
        target_inconsistency=TARGET_INCONSISTENCY,
        utility_type=UTILITY_TYPE,
        prob_model=PROB_MODEL,
        num_intervals=NUM_INTERVALS,
        plot_utilities=PLOT_UTILITIES
    )
    
    gen.verify_consistency(F1_LIST, F2_LIST, F3_LIST)

