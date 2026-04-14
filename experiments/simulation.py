import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from common import utils
from mcda.models import PiecewiseLinearTransformer
from inference.engine import PreferenceSampler

def get_sampler_state(sampler, alg, use_mh_sampler=False):
    """
    Returns the state of the sampler (Samples for BAYES, MAP for FTRL).
    """
    algo_type, model_type = alg.split('-')
    if algo_type == 'BAYES':
        if use_mh_sampler and model_type == 'LIN':
            return sampler.run_mh_sampler(model=model_type)
        else:
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
    overwrite=False,
    calculate_heuristic=False,
    generate_scatter_plots=False,
    table_index=0,
    sub_fold="unknown",
    mape_threshold=0.05,
    plot_mape_fit=False,
    n_samples_mc=2000,
    use_linear_approx=False,
    check_passive_algs_completed=False,
    shared_passive_dir=None,
    use_mh_sampler=False,
    disable_tqdm=False,
    progress_dict=None
):
    """
    Runs the active learning simulation for a single Human Model (table).
    
    Workflow:
    1.  **Setup**: Initialize feature transformer and samplers (Passive & Active).
    2.  **Restoration**: Check for existing progress to resume interrupted runs.
    3.  **Simulation Loop**:
        -   **Passive Track**: Updates a 'Passive' sampler with the ground truth preference for comparison.
        -   **Active Track**:
            -   Calculates Heuristics (Correlation, Stats) if requested.
            -   Suggests the next pair using the Active Strategy (e.g., BALD, US).
            -   **Consistency Check**: Checks if the suggested pair exists in the ground truth data.
                -   If YES: Use the stored preference.
                -   If NO (Sparse Data): Simulate the preference using the true rankings/utility (with prob. consistency).
            -   Updates the 'Active' sampler.
    
    Args:
        table (np.array): The dataset table (items x features).
        ground_truth_prefs (list): List of pairs [winner, loser] representing the user's behavior.
        true_ranking (np.array): Ground truth ranking of items.
        true_utility (np.array): Ground truth utility values (for synthetic consistency check).
        num_steps (int): Number of active learning steps to run.
        ...
    """
    # 1. Setup
    transformer = PiecewiseLinearTransformer.from_equal_intervals(table, num_intervals=3)
    feature_matrix = transformer.transform(table)
    
    sampler_passive = PreferenceSampler(feature_matrix, [], transformer.total_params)
    sampler_active = PreferenceSampler(feature_matrix, [], transformer.total_params)
    
    active_pref_history = [] 
    avg_mi_history = []
    switched_to_us = False

    w_active = None
    path_active_hist = os.path.join(output_dir, "active_prefs.npy")
    path_avg_mi = os.path.join(output_dir, "active_avg_mi.npy")

    # Initialize Shared Passive Folder if reusing
    if check_passive_algs_completed and shared_passive_dir:
        # We need a sub-folder for this table inside the shared dir
        shared_table_dir = os.path.join(shared_passive_dir, f"table_{table_index}")
        utils.save_path(shared_table_dir)
    else:
        shared_table_dir = None

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
            
            if os.path.exists(path_heuristic):
                heuristic_history = np.load(path_heuristic).tolist()

        except Exception:
            # Fallback to fresh start on corruption
            active_pref_history = []
            heuristic_history = []
            w_active = None
            sampler_active = PreferenceSampler(feature_matrix, [], transformer.total_params)

    # 3. Main Loop
    # Build lookup for consistency: frozenset({a,b}) -> [Winner, Loser]
    pref_lookup = {frozenset(pair): pair for pair in ground_truth_prefs}
    all_indices = np.arange(len(table))

    for j in tqdm(range(1, num_steps + 1), desc=f"  Table {table_index}", leave=False, disable=disable_tqdm):
        if progress_dict is not None:
            progress_dict[table_index] = j
        # Determine paths
        if check_passive_algs_completed and shared_table_dir:
            path_passive = os.path.join(shared_table_dir, f"{j}.npy")
        else:
            path_passive = os.path.join(output_dir, f"{j}.npy")

        path_active = os.path.join(output_dir, f"{j}_active.npy")

        # Skip if done
        if os.path.exists(path_passive) and os.path.exists(path_active) and not overwrite:
            # Load passive state to keep sampler consistent? 
            # Ideally we don't need to load if we don't use it, but sampler_passive needs history updates
            # to be correct for FUTURE steps if we were to continue.
            # However, for pure skipping, we just need w_active.
            w_active = np.load(path_active)
            
            # Update samplers to current state for consistency
            p_pair = ground_truth_prefs[j-1]
            sampler_passive.add_preference(p_pair[0], p_pair[1])
            continue
        
        # --- Track A: Passive ---
        passive_pair = ground_truth_prefs[j-1]
        sampler_passive.add_preference(passive_pair[0], passive_pair[1])
        
        # Optimization: Only run passive inference if file doesn't exist
        if os.path.exists(path_passive) and not overwrite:
             w_passive = np.load(path_passive)
        else:
             w_passive = get_sampler_state(sampler_passive, alg, use_mh_sampler=use_mh_sampler)
             np.save(path_passive, w_passive)

        # --- Track B: Active ---
        effective_method = active_method
        
        # Trend Analysis for BALD+US
        if active_method == 'BALD+US' and w_active is not None:
            # 1. Calculate Average MI for ALL pairs
            try:
                mean_mi = sampler_active.calculate_all_pairs_mi(
                    all_indices, 
                    alg, 
                    w_active, 
                    n_samples_mc=n_samples_mc,
                    use_linear_approx=use_linear_approx
                )
                avg_mi_history.append(mean_mi)
                
                # 2. Check for Strategy Switch
                if not switched_to_us and len(avg_mi_history) >= 3:
                    # Fit f(t) = 1/(a+t)
                    t_vals = np.arange(1, len(avg_mi_history) + 1)
                    mi_vals = np.array(avg_mi_history)
                    
                    def decay_func(t, a):
                        return 1.0 / (a + t)
                    
                    # Initial guess for a: a = 1/MI_1 - 1
                    # Avoid division by zero if MI is very small
                    a0 = 1.0 / max(mi_vals[0], 1e-9) - 1.0
                    
                    try:
                        popt, _ = curve_fit(decay_func, t_vals, mi_vals, p0=[a0])
                        a_fit = popt[0]
                        mi_fit = decay_func(t_vals, a_fit)
                        
                        # Calculate MAPE
                        mape = np.mean(np.abs((mi_vals - mi_fit) / mi_vals))
                        
                        just_switched = False
                        if mape > mape_threshold:
                            switched_to_us = True
                            just_switched = True
                        
                        # Diagnostic Plotting
                        if plot_mape_fit and (j % 10 == 0 or just_switched):
                            mape_dir = os.path.join("plots_analysis", "mape_fits", sub_fold, f"table_{table_index}")
                            os.makedirs(mape_dir, exist_ok=True)
                            
                            plt.figure(figsize=(8, 5))
                            plt.plot(t_vals, mi_vals, 'o-', label='Observed Avg MI')
                            plt.plot(t_vals, mi_fit, '--', label=f'Fit: 1/({a_fit:.2f}+t)')
                            plt.title(f"Step {j} | MAPE: {mape:.4f} | Switched: {switched_to_us}")
                            plt.xlabel("Step (t)")
                            plt.ylabel("Avg MI")
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            
                            filename = f"step_{j}.png"
                            if just_switched:
                                filename = f"step_{j}_SWITCH.png"
                                
                            plt.savefig(os.path.join(mape_dir, filename))
                            plt.close()
                            
                    except Exception as e:
                        # Log fit failure but don't crash
                        pass
                
            except Exception as e:
                print(f"MI calculation failed at step {j}: {e}")

        # Determine effective method for suggestion
        if active_method == 'BALD+US':
            effective_method = 'US' if switched_to_us else 'BALD'

        if j == 1:
            active_pair = ground_truth_prefs[0]
            sampler_active.add_preference(active_pair[0], active_pair[1])
            active_pref_history.append(active_pair)
            w_active = np.copy(w_passive)
        else:
            if w_active is None:
                raise ValueError(f"w_active missing at step {j}")

            suggested_pair = sampler_active.suggest_next_pair(
                all_indices, 
                alg=alg, 
                active_method=effective_method, 
                current_state=w_active,
                n_samples_mc=n_samples_mc,
                use_linear_approx=use_linear_approx
            )
            
            if suggested_pair is None:
                suggested_pair = ground_truth_prefs[j-1]

            # --- Consistency Check & Response Simulation ---
            pair_key = frozenset(suggested_pair)
            
            if pair_key in pref_lookup:
                # Case 1: Consistent with Data
                # The suggested pair exists in the pre-generated dataset (ground truth).
                # We simply use the stored decision.
                final_pair = pref_lookup[pair_key]
            else:
                # Case 2: Sparse Data Fallback
                # The suggested pair is NOT in the dataset (common when F3 < 100% or active learning picks obscure pairs).
                # We must SIMULATE the user's response on the fly.
                
                # A. Determine Ground Truth Winner based on Rank
                rank_a = true_ranking[suggested_pair[0]]
                rank_b = true_ranking[suggested_pair[1]]
                
                if rank_a < rank_b:
                    correct_pair = np.array([suggested_pair[0], suggested_pair[1]])
                else:
                    correct_pair = np.array([suggested_pair[1], suggested_pair[0]])
                
                # B. Simulate Human Inconsistency (Softmax)
                # If we have ground truth utility, we apply the Luce-Shepard rule (Softmax) via 'lam'.
                # Ideally, this matches the generation process.
                if true_utility is not None:
                    if np.random.rand() > get_consistency_prob(correct_pair, true_utility, lam):
                        # User makes a mistake (flips preference)
                        final_pair = np.flip(correct_pair)
                    else:
                        final_pair = correct_pair
                else:
                    # Fallback if no utility provided: Assume deterministically correct
                    final_pair = correct_pair

            sampler_active.add_preference(final_pair[0], final_pair[1])
            active_pref_history.append([final_pair[0], final_pair[1]])
            w_active = get_sampler_state(sampler_active, alg, use_mh_sampler=use_mh_sampler)

        np.save(path_active, w_active)
        np.save(path_active_hist, np.array(active_pref_history))
        if active_method == 'BALD+US':
            np.save(path_avg_mi, np.array(avg_mi_history))