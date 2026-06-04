import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from common import utils
from mcda.models import PiecewiseLinearTransformer
from inference.engine import PreferenceSampler

def get_sampler_state(sampler, alg, use_mh_sampler=False, use_hmc_sampler=False, n_samples=2000):
    """
    Returns the state of the sampler (Samples for BAYES, MAP for FTRL).
    """
    algo_type, model_type = alg.split('-')
    if algo_type == 'BAYES':
        if use_hmc_sampler and model_type == 'BT':
            return sampler.run_hmc_sampler(model=model_type, n_samples=n_samples)
        elif use_mh_sampler and model_type == 'LIN':
            return sampler.run_mh_sampler(model=model_type, n_samples=n_samples)
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
    dataset_fold="unknown",
    mape_threshold=0.05,
    plot_mape_fit=False,
    n_samples_mc=2000,
    use_linear_approx=False,
    use_is_bald=False,
    check_passive_algs_completed=False,
    shared_passive_dir=None,
    use_mh_sampler=False,
    use_hmc_sampler=False,
    disable_tqdm=False,
    progress_dict=None,
    n_samples_mcmc=2000):
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

    sum_Delta = 0.0
    sum_V = 0.0
    max_c_t = 0.0

    sum_Delta_all = 0.0
    sum_B_all = 0.0
    error_csv_path = os.path.join(output_dir, "error_scores.csv")
    if overwrite or not os.path.exists(error_csv_path):
        with open(error_csv_path, 'w') as f:
            f.write("step,E_link_rel,E_est_FTLT,epsilon_t,C_t_size,rho_t,E_size,E_sep,j_star,hessian_cond,clipping_used,h_or_scaling,D_j\n")

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
             w_passive = get_sampler_state(sampler_passive, alg, use_mh_sampler=use_mh_sampler, use_hmc_sampler=use_hmc_sampler, n_samples=n_samples_mcmc)
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

            if active_method == 'BALD+US' and not switched_to_us:
                try:
                    candidates, bald_scores = sampler_active.get_candidate_scores(
                        all_indices, alg, 'BALD', w_active, n_samples_mc, use_linear_approx, use_is_bald
                    )
                    
                    mean_mi = np.mean(bald_scores) if candidates else 0.0
                    avg_mi_history.append(mean_mi)
                except Exception as e:
                    print(f"MI calculation failed at step {j}: {e}")
                    candidates = []
                    bald_scores = []

                if not switched_to_us:
                    suggested_pair = candidates[np.argmax(bald_scores)] if candidates else ground_truth_prefs[j-1]
                else:
                    candidates, us_scores = sampler_active.get_candidate_scores(
                        all_indices, alg, 'US', w_active, n_samples_mc, use_linear_approx, use_is_bald
                    )
                    suggested_pair = candidates[np.argmax(us_scores)] if candidates else ground_truth_prefs[j-1]

            elif active_method == 'BALD+US' and switched_to_us:
                candidates, scores = sampler_active.get_candidate_scores(
                    all_indices, alg, 'US', w_active, n_samples_mc, use_linear_approx, use_is_bald
                )
                suggested_pair = candidates[np.argmax(scores)] if candidates else ground_truth_prefs[j-1]

            else:
                candidates, scores = sampler_active.get_candidate_scores(
                    all_indices, alg, active_method, w_active, n_samples_mc, use_linear_approx, use_is_bald
                )
                if active_method == 'BALD':
                    mean_mi = np.mean(scores) if candidates else 0.0
                    avg_mi_history.append(mean_mi)
                suggested_pair = candidates[np.argmax(scores)] if candidates else ground_truth_prefs[j-1]

            # --- Diagnostic Computation for BAYES and FTRL ---
            step_diag_info = None
            if w_active is not None and len(w_active) > 0:
                if alg in ['BAYES-LIN', 'BAYES-BT']:
                    try:
                        samples = w_active
                        idx_a, idx_b = suggested_pair[0], suggested_pair[1]
                        vec_diff = feature_matrix[idx_a] - feature_matrix[idx_b]
                        u_diff = vec_diff @ samples.T
                        
                        if alg == 'BAYES-LIN':
                            probs_1 = 0.5 * (1.0 + u_diff)
                        else:
                            from scipy.special import expit
                            probs_1 = expit(u_diff)
                            
                        probs_1 = np.clip(probs_1, 1e-9, 1.0 - 1e-9)
                        probs_0 = 1.0 - probs_1
                        
                        p_t_1 = np.mean(probs_1)
                        p_t_0 = np.mean(probs_0)
                        
                        a_m_1 = probs_1 / np.sum(probs_1)
                        a_m_0 = probs_0 / np.sum(probs_0)
                        
                        S_t_1 = np.sum(a_m_1 * np.log(probs_1 / p_t_1))
                        S_t_0 = np.sum(a_m_0 * np.log(probs_0 / p_t_0))
                        
                        B_t = p_t_1 * S_t_1 + p_t_0 * S_t_0
                        
                        step_diag_info = {
                            'B_t': B_t,
                            'S_t_1': S_t_1,
                            'S_t_0': S_t_0,
                            'p_t_1': p_t_1,
                            'p_t_0': p_t_0
                        }
                    except Exception as e:
                        print(f"BAYES Diagnostic computation failed: {e}")
                        
                elif alg in ['FTRL-LIN', 'FTRL-BT']:
                    try:
                        # B_t (model-expected Bayesian surprise) is computed below, AFTER the
                        # hypothetical-answer surprises S_t(0), S_t(1), as their predictive average.
                        # This keeps it consistent with the realized surprise S_t(y_t), so that the
                        # link residual Delta_t = S_t(y_t) - B_t is a proper martingale difference
                        # (zero conditional mean under correct link specification). Using the
                        # Laplace-Taylor closed-form BALD score here instead would mix two different
                        # approximations and bias the residual.
                        w_hat_prev = w_active
                        Sigma_prev = sampler_active.compute_laplace_covariance(w_hat_prev, alg=alg)
                        
                        # FTRL predictive probability
                        idx_a, idx_b = suggested_pair[0], suggested_pair[1]
                        vec_diff = feature_matrix[idx_a] - feature_matrix[idx_b]
                        u_diff = vec_diff @ w_hat_prev
                        
                        if alg == 'FTRL-LIN':
                            p_t_1 = 0.5 * (1.0 + u_diff)
                        else:
                            from scipy.special import expit
                            p_t_1 = expit(u_diff)
                        
                        p_t_1 = np.clip(p_t_1, 1e-9, 1.0 - 1e-9)
                        p_t_0 = 1.0 - p_t_1
                        
                        # Hypothetical one-step updates for both answers:
                        # Temporarily append hypothetical preference, optimize, and compute covariance
                        original_prefs = sampler_active.prefs.copy()
                        
                        # Hypothetical answer 1 (winner suggested_pair[0], loser suggested_pair[1])
                        sampler_active.add_preference(suggested_pair[0], suggested_pair[1])
                        w_hat_t_1 = get_sampler_state(sampler_active, alg, use_mh_sampler=use_mh_sampler, use_hmc_sampler=use_hmc_sampler)
                        Sigma_t_1 = sampler_active.compute_laplace_covariance(w_hat_t_1, alg=alg)
                        
                        # Rollback
                        sampler_active.prefs = original_prefs.copy()
                        sampler_active._update_diff_vectors()
                        
                        # Hypothetical answer 0 (winner suggested_pair[1], loser suggested_pair[0])
                        sampler_active.add_preference(suggested_pair[1], suggested_pair[0])
                        w_hat_t_0 = get_sampler_state(sampler_active, alg, use_mh_sampler=use_mh_sampler, use_hmc_sampler=use_hmc_sampler)
                        Sigma_t_0 = sampler_active.compute_laplace_covariance(w_hat_t_0, alg=alg)
                        
                        # Rollback
                        sampler_active.prefs = original_prefs.copy()
                        sampler_active._update_diff_vectors()
                        
                        # Extract unconstrained parameters in ALR space for FTRL-LIN
                        if alg == 'FTRL-LIN':
                            mu_prev = np.log(w_hat_prev[:-1] + 1e-12) - np.log(w_hat_prev[-1] + 1e-12)
                            mu_t_1 = np.log(w_hat_t_1[:-1] + 1e-12) - np.log(w_hat_t_1[-1] + 1e-12)
                            mu_t_0 = np.log(w_hat_t_0[:-1] + 1e-12) - np.log(w_hat_t_0[-1] + 1e-12)
                        else:
                            mu_prev = w_hat_prev
                            mu_t_1 = w_hat_t_1
                            mu_t_0 = w_hat_t_0
                        
                        # Compute KL Surprises
                        def gaussian_kl(mu_1, Sigma_1, mu_0, Sigma_0, jitter=1e-8):
                            k = len(mu_1)
                            Sigma_0_reg = Sigma_0 + np.eye(k) * jitter
                            Sigma_1_reg = Sigma_1 + np.eye(k) * jitter
                            try:
                                Sigma_0_inv_Sigma_1 = np.linalg.solve(Sigma_0_reg, Sigma_1_reg)
                                tr_term = np.trace(Sigma_0_inv_Sigma_1)
                                diff = mu_0 - mu_1
                                mean_term = diff.T @ np.linalg.solve(Sigma_0_reg, diff)
                                sign_0, logdet_0 = np.linalg.slogdet(Sigma_0_reg)
                                sign_1, logdet_1 = np.linalg.slogdet(Sigma_1_reg)
                                logdet_term = logdet_0 - logdet_1
                                kl = 0.5 * (tr_term + mean_term - k + logdet_term)
                                return max(0.0, kl)
                            except:
                                return 0.0
                        
                        S_t_1 = gaussian_kl(mu_t_1, Sigma_t_1, mu_prev, Sigma_prev)
                        S_t_0 = gaussian_kl(mu_t_0, Sigma_t_0, mu_prev, Sigma_prev)

                        # Model-expected Bayesian surprise: predictive average of the hypothetical
                        # Gaussian-KL surprises, weighted by the MAP plug-in predictive probabilities
                        # (p_t_1, p_t_0 computed from w_hat_prev above). This is the Laplace analogue
                        # of B_t = sum_y p_hat_MAP(y|q_t) S_tilde_t(y) in the paper, and matches the
                        # BAYES branch (B_t = p_t_1*S_t_1 + p_t_0*S_t_0), so Delta_t = S_t - B_t is a
                        # consistent link residual.
                        B_t = p_t_1 * S_t_1 + p_t_0 * S_t_0

                        step_diag_info = {
                            'B_t': B_t,
                            'S_t_1': S_t_1,
                            'S_t_0': S_t_0,
                            'p_t_1': p_t_1,
                            'p_t_0': p_t_0,
                            'Sigma_prev': Sigma_prev
                        }
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"FTRL Diagnostic computation failed: {e}")

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

            if step_diag_info is not None:
                if list(final_pair) == list(suggested_pair):
                    y_t = 1
                    S_t = step_diag_info['S_t_1']
                else:
                    y_t = 0
                    S_t = step_diag_info['S_t_0']
                
                B_t = step_diag_info['B_t']
                p_t_1 = step_diag_info['p_t_1']
                p_t_0 = step_diag_info['p_t_0']
                S_t_1 = step_diag_info['S_t_1']
                S_t_0 = step_diag_info['S_t_0']

                # Compute Link Error Delta_t
                Delta_t = S_t - B_t
                sum_Delta_all += Delta_t
                sum_B_all += B_t
                E_link_rel = abs(sum_Delta_all) / (sum_B_all + 1e-6)

                # Compute Estimation Error
                E_est_FTLT = 0.0
                epsilon_t = 0.0
                C_t_size = 0
                rho_t = 0.0
                E_size = 0.0
                E_sep = 0.0
                
                j_star = ''
                hessian_cond = ''
                clipping_used = ''
                h_or_scaling = ''
                D_j_str = ''

                if alg in ['FTRL-LIN', 'FTRL-BT']:
                    Sigma_prev = step_diag_info.get('Sigma_prev', None)
                    if Sigma_prev is not None:
                        est_diag = sampler_active.calculate_estimation_error_diagnostic(w_active, Sigma_prev, [suggested_pair], alg)
                        epsilon_t = est_diag['epsilon_t']
                        E_est_FTLT = est_diag['cand_errors'].get(tuple(suggested_pair), np.inf)
                        
                        j_star = est_diag.get('j_star', '')
                        hessian_cond = est_diag.get('hessian_cond', '')
                        clipping_used = est_diag.get('clipping_used', '')
                        if alg == 'FTRL-BT':
                            h_or_scaling = est_diag.get('bt_scaling', '')
                        else:
                            h_or_scaling = est_diag.get('lin_h', '')
                        
                        D_j_val = est_diag.get('D_j', None)
                        if D_j_val is not None:
                            import json
                            D_j_str = '"' + json.dumps(D_j_val.tolist()) + '"'
                            
                elif alg in ['BAYES-LIN', 'BAYES-BT']:
                    model_type = alg.split('-')[1]
                    bayes_est = sampler_active.calculate_bayes_estimation_error(candidates, w_active, model_type, active_method)
                    C_t_size = bayes_est['|C_t|']
                    rho_t = bayes_est['rho_t']
                    E_size = bayes_est['Delta_MC']
                    E_sep = bayes_est['mcse_max']

                # Log to CSV
                with open(error_csv_path, 'a') as ef:
                    ef.write(f"{j},{E_link_rel},{E_est_FTLT},{epsilon_t},{C_t_size},{rho_t},{E_size},{E_sep},{j_star},{hessian_cond},{clipping_used},{h_or_scaling},{D_j_str}\n")

                if active_method == 'BALD+US' and not switched_to_us:
                    V_t = p_t_1 * (S_t_1 - B_t)**2 + p_t_0 * (S_t_0 - B_t)**2
                    
                    sum_Delta += Delta_t
                    sum_V += V_t
                    c_t = max(abs(S_t_1 - B_t), abs(S_t_0 - B_t))
                    max_c_t = max(max_c_t, c_t)
                    
                    G_t = -sum_Delta
                    if G_t > 0:
                        p_mart = np.exp(-G_t**2 / (2.0 * (sum_V + max_c_t * G_t / 3.0)))
                        if p_mart < 0.05:
                            switched_to_us = True
                

            sampler_active.add_preference(final_pair[0], final_pair[1])
            active_pref_history.append([final_pair[0], final_pair[1]])
            w_active = get_sampler_state(sampler_active, alg, use_mh_sampler=use_mh_sampler, use_hmc_sampler=use_hmc_sampler, n_samples=n_samples_mcmc)

        np.save(path_active, w_active)
        np.save(path_active_hist, np.array(active_pref_history))
        if active_method in ['BALD+US', 'BALD']:
            np.save(path_avg_mi, np.array(avg_mi_history))
