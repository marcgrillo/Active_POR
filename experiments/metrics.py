import os
import numpy as np
from tqdm import tqdm
from common.utils import save_path, read_dataset
from mcda.models import PiecewiseLinearTransformer

class BenchmarkRunner:
    """
    Orchestrates batch experiments for POI, RAI, AIOS, ASPS, ASRS.
    """

    def __init__(self, dataset_fold, sub_fold, num_subint, hm, F1, F2, F3, num_dm_dec):
        self.dataset_fold = dataset_fold
        self.sub_fold = sub_fold
        self.samples_fold = f'samples_{dataset_fold}'
        self.tests_fold = f'tests_{dataset_fold}'
        self.sub_tests_fold = os.path.join(self.tests_fold, sub_fold)
        
        self.num_subint = num_subint
        self.hm = hm # Human models / runs
        self.num_dm_dec = num_dm_dec
        self.F1, self.F2, self.F3 = F1, F2, F3
        
        save_path(self.sub_tests_fold)

    def _sample_path(self, f1, f2, f3, i, j, active=False):
        """Construct path to sample files."""
        suffix = "_active.npy" if active else ".npy"
        return os.path.join(
            self.samples_fold,
            f"f1_{f1}_f2_{f2}_f3_{f3}",
            self.sub_fold,
            f"table_{i}",
            f"{j}{suffix}",
        )

    # ----------------------------------------------------------------------
    # Metrics (POI / RAI)
    # ----------------------------------------------------------------------

    def calc_poi(self, f1, t, sam):
        """Preference Ordering Index: pairwise dominance probabilities."""
        rank_count = np.zeros((f1, f1))
        sam = np.atleast_2d(sam)
        vals = t @ sam.T 
        
        for col in range(vals.shape[1]):
            scores = vals[:, col]
            rank_count += (scores[:, None] >= scores[None, :])
            
        return rank_count / vals.shape[1]

    def calc_rai(self, f1, t, sam):
        """Rank Acceptability Index: distribution of rank positions."""
        rank_count = np.zeros((f1, f1))
        sam = np.atleast_2d(sam)
        vals = t @ sam.T
        
        for col in range(vals.shape[1]):
            scores = vals[:, col]
            ranks = np.argsort(np.argsort(-scores))
            rank_count[np.arange(f1), ranks] += 1
            
        return rank_count / vals.shape[1]

    # ----------------------------------------------------------------------
    # Main Computation Loop
    # ----------------------------------------------------------------------

    def compute_metrics(self, metric_type="poi", force=False):
        """Generic loop to compute POI or RAI and save to disk."""
        metric_name = "pois" if metric_type == "poi" else "rais"
        func = self.calc_poi if metric_type == "poi" else self.calc_rai
        
        out_fold = os.path.join(self.sub_tests_fold, metric_name)
        save_path(out_fold)
        print(f"Calculating {metric_name}...")

        for f1 in tqdm(self.F1):
            for f2 in self.F2:
                for f3 in self.F3:
                    tables, _, _, _ = read_dataset(self.dataset_fold, f1, f2, f3)
                    
                    for i in range(self.hm):
                        table_dir = os.path.join(out_fold, f"f1_{f1}_f2_{f2}_f3_{f3}", f"table_{i}")
                        save_path(table_dir)

                        transformer = PiecewiseLinearTransformer.from_equal_intervals(tables[i], self.num_subint)
                        t_vec = transformer.transform(tables[i])

                        for j in range(1, self.num_dm_dec + 1):
                            path_std = os.path.join(table_dir, f"{j}.npy")
                            path_act = os.path.join(table_dir, f"{j}_active.npy")

                            if os.path.exists(path_std) and not force:
                                continue

                            try:
                                sam = np.load(self._sample_path(f1, f2, f3, i, j, False))
                                np.save(path_std, func(f1, t_vec, sam))

                                sam_act = np.load(self._sample_path(f1, f2, f3, i, j, True))
                                np.save(path_act, func(f1, t_vec, sam_act))
                            except FileNotFoundError:
                                pass

    # ----------------------------------------------------------------------
    # Aggregated Indices (ASRS, ASPS, AIOS)
    # ----------------------------------------------------------------------
    
    def compute_aios(self, force=False):
        """Average Individual Ordinal Stability (Stability of Best Alternative)."""
        test_name = "aios"
        out_fold = os.path.join(self.sub_tests_fold, test_name)
        rais_fold = os.path.join(self.sub_tests_fold, "rais")
        save_path(out_fold)

        print(f"Calculating {test_name}...")
        for f1 in tqdm(self.F1):
            for f2 in self.F2:
                for f3 in self.F3:
                    _, rankings, _, _ = read_dataset(self.dataset_fold, f1, f2, f3)
                    sub_out = os.path.join(out_fold, f"f1_{f1}_f2_{f2}_f3_{f3}")
                    save_path(sub_out)

                    for j in range(1, self.num_dm_dec + 1):
                        p_std = os.path.join(sub_out, f"{test_name}_{j}.npy")
                        p_act = os.path.join(sub_out, f"{test_name}_active_{j}.npy")

                        if os.path.exists(p_std) and not force: continue

                        res_std, res_act = [], []
                        for i in range(self.hm):
                            try:
                                r_path = os.path.join(rais_fold, f"f1_{f1}_f2_{f2}_f3_{f3}", f"table_{i}", f"{j}.npy")
                                rai = np.load(r_path)
                                rai_act = np.load(r_path.replace(".npy", "_active.npy"))
                                
                                best_idx = np.argwhere(rankings[i] == 0)[0, 0]
                                res_std.append(rai[best_idx, 0])
                                res_act.append(rai_act[best_idx, 0])
                            except (FileNotFoundError, IndexError):
                                break
                        
                        np.save(p_std, res_std)
                        np.save(p_act, res_act)

    def compute_asps(self, force=False):
        """Average Simulated Preference Stability (Stability of Pairwise Relations)."""
        test_name = "asps"
        out_fold = os.path.join(self.sub_tests_fold, test_name)
        pois_fold = os.path.join(self.sub_tests_fold, "pois")
        save_path(out_fold)

        print(f"Calculating {test_name}...")
        for f1 in tqdm(self.F1):
            for f2 in self.F2:
                for f3 in self.F3:
                    _, rankings, _, _ = read_dataset(self.dataset_fold, f1, f2, f3)
                    sub_out = os.path.join(out_fold, f"f1_{f1}_f2_{f2}_f3_{f3}")
                    save_path(sub_out)

                    for j in range(1, self.num_dm_dec + 1):
                        p_std = os.path.join(sub_out, f"{test_name}_{j}.npy")
                        # ASPS saves active in same file logic or separate? Usually separate.
                        # Original gentests used same logic.
                        if os.path.exists(p_std) and not force: continue

                        res_std, res_act = [], []
                        for i in range(self.hm):
                            try:
                                poi_path = os.path.join(pois_fold, f"f1_{f1}_f2_{f2}_f3_{f3}", f"table_{i}", f"{j}.npy")
                                poi = np.load(poi_path)
                                poi_act = np.load(poi_path.replace(".npy", "_active.npy"))

                                pref_mask = rankings[i][:, None] > rankings[i][None, :]
                                coef = 2 / (f1 * (f1 - 1))
                                res_std.append(np.sum(poi.T * pref_mask) * coef)
                                res_act.append(np.sum(poi_act.T * pref_mask) * coef)
                            except FileNotFoundError:
                                break
                        
                        np.save(p_std, res_std)
                        np.save(p_std.replace(".npy", "_active.npy"), res_act)

    def compute_asrs(self, force=False):
        """Average Simulated Rank Stability (Average probability of correct rank assignment)."""
        test_name = "asrs"
        out_fold = os.path.join(self.sub_tests_fold, test_name)
        rais_fold = os.path.join(self.sub_tests_fold, "rais")
        save_path(out_fold)

        print(f"Calculating {test_name}...")
        for f1 in tqdm(self.F1):
            for f2 in self.F2:
                for f3 in self.F3:
                    _, rankings, _, _ = read_dataset(self.dataset_fold, f1, f2, f3)
                    sub_out = os.path.join(out_fold, f"f1_{f1}_f2_{f2}_f3_{f3}")
                    save_path(sub_out)

                    for j in range(1, self.num_dm_dec + 1):
                        p_std = os.path.join(sub_out, f"{test_name}_{j}.npy")
                        p_act = os.path.join(sub_out, f"{test_name}_active_{j}.npy") # Explicit separate path for consistency

                        if os.path.exists(p_std) and not force: continue

                        res_std, res_act = [], []
                        for i in range(self.hm):
                            try:
                                r_path = os.path.join(rais_fold, f"f1_{f1}_f2_{f2}_f3_{f3}", f"table_{i}", f"{j}.npy")
                                rai = np.load(r_path)
                                rai_act = np.load(r_path.replace(".npy", "_active.npy"))
                                
                                # ASRS Logic: Mean probability that alt 'k' is at rank 'k'
                                # We sum the diagonal of the RAI matrix re-ordered by true rank
                                # OR simpler: for each rank k (0..f1-1), find which alt has that true rank
                                # and get its probability of being at rank k.
                                current_rank = rankings[i]
                                
                                # List comprehension for mean calculation
                                # rai[alt_index, rank_index]
                                val_std = np.mean([
                                    rai[np.argwhere(current_rank == k)[0, 0], k] 
                                    for k in range(f1)
                                ])
                                
                                val_act = np.mean([
                                    rai_act[np.argwhere(current_rank == k)[0, 0], k] 
                                    for k in range(f1)
                                ])
                                
                                res_std.append(val_std)
                                res_act.append(val_act)
                            except (FileNotFoundError, IndexError):
                                break
                        
                        np.save(p_std, res_std)
                        np.save(p_act, res_act)