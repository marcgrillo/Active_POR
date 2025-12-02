import os
import numpy as np
from tqdm import tqdm
from mcda.models import PiecewiseLinearTransformer
from inference.engine import PreferenceSampler

def get_sampler_state(sampler, alg):
    """
    Returns the state of the sampler (Samples for BAYES, MAP for FTRL).
    """
    algo_type, model_type = alg.split('-')
    if algo_type == 'BAYES':
        return sampler.run_nested(model=model_type, nlive=500)
    elif algo_type == 'FTRL':
        return sampler.find_map(model=model_type)
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")

def get_consistency_prob(pair, U, lam):
    """Calculates P(correct) based on utility difference."""
    u_winner = U[pair[0]]
    u_loser  = U[pair[1]]
    diff = u_winner - u_loser
    return 1.0 / (1.0 + np.round(np.exp(-lam * diff), 10))

def process_single_table(
    table, 
    ground_truth_prefs, 
    true_ranking,
    true_utility,
    num_steps, 
    output_dir, 
    alg='BAYES-BT', 
    active_method='BALD',
    lam=10.0,
    overwrite=False
):
    """
    Runs simulation for one Human Model.
    Resumes from the last saved step if files exist.
    """
    # 1. Setup
    transformer = PiecewiseLinearTransformer.from_equal_intervals(table, num_intervals=3)
    feature_matrix = transformer.transform(table)
    
    sampler_passive = PreferenceSampler(feature_matrix, [], transformer.total_params)
    sampler_active = PreferenceSampler(feature_matrix, [], transformer.total_params)
    
    active_pref_history = [] 
    w_active = None 
    path_active_hist = os.path.join(output_dir, "active_prefs.npy")

    # 2. Attempt Restoration
    if os.path.exists(path_active_hist) and not overwrite:
        try:
            active_pref_history = np.load(path_active_hist).tolist()
            active_pref_history = [list(p) for p in active_pref_history]
            for pair in active_pref_history:
                sampler_active.add_preference(pair[0], pair[1])
            
            # Find last valid w_active
            for k in range(num_steps, 0, -1):
                path_last_w = os.path.join(output_dir, f"{k}_active.npy")
                if os.path.exists(path_last_w):
                    w_active = np.load(path_last_w)
                    break
        except Exception:
            # Fallback to fresh start on corruption
            active_pref_history = []
            w_active = None
            sampler_active = PreferenceSampler(feature_matrix, [], transformer.total_params)

    # 3. Main Loop
    for j in tqdm(range(1, num_steps + 1), desc="  Constraints", leave=False):
        path_passive = os.path.join(output_dir, f"{j}.npy")
        path_active = os.path.join(output_dir, f"{j}_active.npy")

        # Skip if done
        if os.path.exists(path_passive) and os.path.exists(path_active) and not overwrite:
            w_active = np.load(path_active)
            p_pair = ground_truth_prefs[j-1]
            sampler_passive.add_preference(p_pair[0], p_pair[1])
            continue

        # --- Track A: Passive ---
        passive_pair = ground_truth_prefs[j-1]
        sampler_passive.add_preference(passive_pair[0], passive_pair[1])
        w_passive = get_sampler_state(sampler_passive, alg)
        np.save(path_passive, w_passive)

        # --- Track B: Active ---
        if j == 1:
            active_pair = ground_truth_prefs[0]
            sampler_active.add_preference(active_pair[0], active_pair[1])
            active_pref_history.append(active_pair)
            w_active = np.copy(w_passive)
        else:
            if w_active is None:
                raise ValueError(f"w_active missing at step {j}")

            all_indices = np.arange(len(table))
            suggested_pair = sampler_active.suggest_next_pair(
                all_indices, 
                alg=alg, 
                active_method=active_method, 
                current_state=w_active,
                n_samples_mc=200
            )
            
            if suggested_pair is None:
                suggested_pair = ground_truth_prefs[j-1]

            # Determine Correct Order
            rank_a = true_ranking[suggested_pair[0]]
            rank_b = true_ranking[suggested_pair[1]]
            
            if rank_a < rank_b:
                correct_pair = np.array([suggested_pair[0], suggested_pair[1]])
            else:
                correct_pair = np.array([suggested_pair[1], suggested_pair[0]])
            
            # Simulate Inconsistency
            if true_utility is not None:
                if np.random.rand() > get_consistency_prob(correct_pair, true_utility, lam):
                    final_pair = np.flip(correct_pair)
                else:
                    final_pair = correct_pair
            else:
                final_pair = correct_pair

            sampler_active.add_preference(final_pair[0], final_pair[1])
            active_pref_history.append([final_pair[0], final_pair[1]])
            w_active = get_sampler_state(sampler_active, alg)

        np.save(path_active, w_active)
        np.save(path_active_hist, np.array(active_pref_history))