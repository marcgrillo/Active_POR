import os
import json
import numpy as np
from scipy.stats import dirichlet
from scipy.optimize import brentq
from tqdm import tqdm
from common import utils

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
        return np.sum(u_j, axis=1)

    def _calculate_expected_inconsistency(self, lam, utility_diffs):
        """P(flip) = 1 - sigmoid(lambda * delta_U)"""
        probs = 1.0 - utils.robust_sigmoid(lam * utility_diffs)
        return np.mean(probs)

    def _tune_lambda(self, f1, f2, target_inconsistency):
        """Finds the lambda value that results in the desired inconsistency rate."""
        if target_inconsistency <= 0: return 1000.0
        if target_inconsistency >= 0.5: return 0.0

        # 1. Calibration set of Utility Differences
        calibration_size = 1000
        diffs = []
        for _ in range(calibration_size):
            ls = np.random.rand(f1, f2)
            U = self._generate_exponential_utility(ls, f2)
            idx = np.arange(f1)
            np.random.shuffle(idx)
            u_shuffled = U[idx]
            d = np.abs(u_shuffled[:-1] - u_shuffled[1:])
            diffs.append(d)
        
        utility_diffs = np.concatenate(diffs)

        # 2. Root finding
        def objective(lam):
            return self._calculate_expected_inconsistency(lam, utility_diffs) - target_inconsistency

        try:
            y_min = objective(0.01)
            y_max = objective(100.0)
            if y_min * y_max > 0:
                return 0.01 if abs(y_min) < abs(y_max) else 100.0
            return brentq(objective, 0.01, 100.0)
        except ValueError:
            return 10.0

    def generate_batch(self, F1, F2, F3, target_inconsistency=0.0):
        print(f"Generating datasets in '{self.output_dir}' with {target_inconsistency*100:.1f}% inconsistency...")
        
        # Registry to store lambda values for benchmarking
        params_registry = {
            "target_inconsistency": target_inconsistency
        }
        
        for f1 in tqdm(F1, desc="Alternatives (F1)"):
            all_couples = utils.get_combinations(np.arange(f1))
            indices = np.arange(len(all_couples))
            
            for f2 in F2:
                # --- Step 1: Auto-tune Lambda ---
                if target_inconsistency > 0:
                    lam = self._tune_lambda(f1, f2, target_inconsistency)
                else:
                    lam = int(1e9)  # Effectively no inconsistency
                
                # Save lambda for this configuration
                config_key = f"f1_{f1}_f2_{f2}"
                params_registry[config_key] = lam

                for f3 in F3:
                    num_dm_dec = int(np.round(f3 * len(all_couples) / 100))
                    
                    lss_batch, ranks_prefs_batch, Us_batch = [], [], []
                    
                    for _ in range(self.n_runs):
                        ls = np.random.rand(f1, f2)
                        U = self._generate_exponential_utility(ls, f2)
                        rk = self._compute_rank(U)
                        
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
                            threshold = utils.robust_sigmoid(lam * u_diff)
                            
                            if np.random.rand() > threshold:
                                sorted_couples.append([loser, winner]) # Swap
                            else:
                                sorted_couples.append([winner, loser]) # Correct
                            
                        lss_batch.append(ls)
                        Us_batch.append(U)
                        flat_pairs = np.hstack(sorted_couples)
                        row_data = np.hstack([rk, flat_pairs])
                        ranks_prefs_batch.append(row_data)
                    
                    base_name = f"f1_{f1}__f2_{f2}__ndm_{num_dm_dec}"
                    np.savetxt(os.path.join(self.output_dir, f"{base_name}table.csv"), np.vstack(lss_batch))
                    np.savetxt(os.path.join(self.output_dir, f"{base_name}rank+preferences.csv"), np.vstack(ranks_prefs_batch))
                    np.savetxt(os.path.join(self.output_dir, f"{base_name}Us.csv"), np.vstack(Us_batch))

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
    F1 = [30]
    F2 = [4]
    F3 = [25]
    
    # Example: 10% inconsistency
    gen = DatasetGenerator(output_dir="datasets", n_runs=10)
    gen.generate_batch(F1, F2, F3, target_inconsistency=0.0)
    gen.verify_consistency(F1, F2, F3)