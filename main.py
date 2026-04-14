import numpy as np
from experiments.runner import run_batch_experiments
from experiments.metrics import BenchmarkRunner
from common.utils import parse_subfold_string
from experiments.plotting import plot_metric_results

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# ==============================================================================
# 1. Experiment Dimensions (Dataset Selection)
# ==============================================================================
# F1: List of integers representing the number of Alternatives (items) in the dataset.
#     The code will iterate over these values to find matching datasets.
F1 = [30]

# F2: List of integers representing the number of Criteria (attributes) per alternative.
F2 = [4]

# F3: List of integers representing the Budget of pairwise comparisons as a percentage 
#     of the total possible unique pairs. E.g., 100 means the budget is enough to 
#     compare 100% of pairs (though the active learner selects which ones).
#     Budget N = round(F3 * (F1 * (F1 - 1) / 200))
F3 = [100]

# DATASET_FOLDS: List of dataset names to process. 
#       The code will look for raw data in: 'datasets/{name}/'
#       The code will save results in: 'samples/{name}/', 'tests/{name}/', and 'figs/{name}/'
DATASET_FOLDS = ['default_dataset']

# ==============================================================================
# 2. Algorithm Configuration
# ==============================================================================
# TARGET_METHODS: List of strings defining which active learning strategies to run.
#     Format: "{MODEL}_{MODE}_{ACQUISITION}" or "{MODEL}_{MODE}_{ACQUISITION}+{EXTRA}"
#     - MODEL: 'BAYES' (Bayesian) or 'FTRL' (Follow The Regularized Leader)
#     - MODE: 'LIN' (Linear utility) or 'BT' (Bradley-Terry probability model)
#     - ACQUISITION: 
#         - 'BALD': Bayesian Active Learning by Disagreement (Mutual Information)
#         - 'US': Uncertainty Sampling
#         - 'PASSIVE': Random selection (usually implicitly handled, not manually set here)
TARGET_METHODS = [
    #'BAYES_LIN_BALD',      # Bayesian Linear Model with BALD
    #'BAYES_BT_BALD',       # Bayesian Bradley-Terry with BALD
    #'FTRL_LIN_BALD',        # FTRL Linear Model with BALD
    #'FTRL_BT_BALD',         # FTRL Bradley-Terry with BALD
    #'FTRL_BT_BALD+US',      # FTRL BT with Hybrid BALD + Uncertainty Sampling
    #'BAYES_LIN_US',        # Bayesian Linear with Uncertainty Sampling
    #'BAYES_BT_US',         # Bayesian BT with Uncertainty Sampling
    #'FTRL_LIN_US',          # FTRL Linear with Uncertainty Sampling
    #'FTRL_BT_US',           # FTRL BT with Uncertainty Sampling
    #'FTRL_LIN_BALD+US',
    'BAYES_LIN_BALD+US',
] 

# ==============================================================================
# 3. Simulation Parameters
# ==============================================================================
# HM_FTRL: Number of "Human Models" to process for FTRL algorithms.
HM_FTRL = 10
# HM_BAYES: Number of "Human Models" to process for BAYES algorithms.
HM_BAYES = 1


# MAPE_THRESHOLD: Used only for 'BALD+US' switching strategy.
#       This is the maximum allowed MAPE for the 1/(a+t) trend-line fit 
#       of average MI scores. If exceeded, the method switches to US.
MAPE_THRESHOLD = 0.05 

# N_SAMPLES_MC: Number of Monte Carlo samples used to approximate the posterior distribution
#       when calculating Mutual Information (BALD), specifically when analytic approximations
#       are not used or not applicable. Higher = more accurate but slower.
N_SAMPLES_MC = 2000 

# USE_LINEAR_MI_APPROX: Boolean flag for BALD optimization.
#       - True: Use a closed-form analytic linear approximation for Mutual Information.
#               Drastically faster but relies on linear assumptions.
#       - False: Use Monte Carlo sampling to estimate MI. Slower but potentially more robust
#                for non-linear scenarios or complex posteriors.
USE_LINEAR_MI_APPROX = False 
# PLOT_MAPE_FIT: If True, saves diagnostic plots of the MI decay fit every 10 steps.
PLOT_MAPE_FIT = True

# CHECK_PASSIVE_ALGS_COMPLETED: Optimization flag.
#       - True: Before running a simulation, check if a 'PASSIVE' (Random) run already 
#               exists for this dataset config. If it does, reuse those results as the 
#               baseline/passive curve instead of re-simulating random selection.
#       - False: Always re-run the passive/random baseline.
CHECK_PASSIVE_ALGS_COMPLETED = True

# USE_MH_SAMPLER: Flag to enable Metropolis-Hastings sampling for the BAYES LIN configuration
USE_MH_SAMPLER = True

# ==============================================================================
# 4. Execution Flow Flags
# ==============================================================================

# OVERWRITE: 
#       - True: Re-run experiments even if output files already exist in the results folder.
#       - False: Skip experiments that have already been completed.
OVERWRITE = False

# CALCULATE_METRICS: 
#       - True: Run the metric calculation phase (e.g., Percent Increase, Accuracy) 
#               after the simulation loop.
CALCULATE_METRICS = True

# FORCE_METRICS: 
#       - True: Re-calculate metrics even if the metric summary files already exist.
#       - False: Skip metric calculation if files exist.
FORCE_METRICS = True

# GENERATE_PLOTS: 
#       - True: Generate visualization plots (line graphs of performance vs. queries) 
#               and save them to the results folders.
GENERATE_PLOTS = True

# CALCULATE_HEURISTIC: 
#       - True: Compute additional heuristic statistics (e.g., correlations between 
#               acquisition scores and actual utility reduction). Useful for debugging 
#               or deep analysis of strategy behavior.
CALCULATE_HEURISTIC = True

# GENERATE_SCATTER_PLOTS: 
#       - True: Generate scatter plots for every step of the simulation showing 
#               parameter estimates vs true utilities. 
#       - WARNING: Very slow and generates a huge number of files. Use only for deep debugging.
GENERATE_SCATTER_PLOTS = False

# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Run Experiments
    for sub_fold in TARGET_METHODS:
        alg_name, active_method_name = parse_subfold_string(sub_fold)
        print(f"\n>>> Running: {alg_name} with {active_method_name}")

        algo_type, _ = alg_name.split('-')
        HM = HM_BAYES if algo_type == 'BAYES' else HM_FTRL
        
        run_batch_experiments(
            F1, F2, F3, 
            sub_fold=sub_fold, 
            dataset_folds=DATASET_FOLDS, 
            alg=alg_name, 
            active_method=active_method_name, 
            overwrite=OVERWRITE,
            hm=HM,  # Pass the limit here
            calculate_heuristic=CALCULATE_HEURISTIC,
            generate_scatter_plots=GENERATE_SCATTER_PLOTS,
            mape_threshold=MAPE_THRESHOLD,
            plot_mape_fit=PLOT_MAPE_FIT,
            n_samples_mc=N_SAMPLES_MC,
            use_linear_approx=USE_LINEAR_MI_APPROX,
            check_passive_algs_completed=CHECK_PASSIVE_ALGS_COMPLETED,
            use_mh_sampler=USE_MH_SAMPLER,
        )
        
    # 2. Calculate Metrics
    force = FORCE_METRICS
    if CALCULATE_METRICS:
        # Note: Using first F1/F2/F3 config for metric calculation setup
        f1, f2, f3 = F1[0], F2[0], F3[0]
        num_dm_dec = int(np.round(f3 * (f1 * (f1 - 1) / 200)))
        
        for sub_fold in TARGET_METHODS:
            alg_name, active_method_name = parse_subfold_string(sub_fold)

            algo_type, _ = alg_name.split('-')
            HM = HM_BAYES if algo_type == 'BAYES' else HM_FTRL

            print(f"\n=== Calculating Metrics for {alg_name} with {active_method_name} ===")
            runner = BenchmarkRunner(
                dataset_fold=DATASET_FOLDS[0],
                sub_fold=sub_fold, # Metrics for first method in list
                num_subint=3,
                hm=HM, # Use same limit here
                F1=F1, F2=F2, F3=F3,
                num_dm_dec=num_dm_dec
            )
            runner.compute_perc_inc(force=force)
            runner.compute_metrics("poi", force=force)
            runner.compute_metrics("rai", force=force)
            runner.compute_asrs(force=force)
            runner.compute_aios(force=force)
            runner.compute_asps(force=force)

    # 3. Generate Plots
    if GENERATE_PLOTS:
        f1, f2, f3 = F1[0], F2[0], F3[0]
        num_dm_dec = int(np.round(f3 * (f1 * (f1 - 1) / 200)))
        
        metrics_to_plot = ['perc_inc', 'asrs', 'asps', 'aios']
        
        for sub_fold in TARGET_METHODS:
            alg_name, active_method_name = parse_subfold_string(sub_fold)
            
            # Determine HM for plotting (should match what was used for metrics)
            algo_type, _ = alg_name.split('-')
            HM = HM_BAYES if algo_type == 'BAYES' else HM_FTRL

            print(f"\n=== Generating Plots for {alg_name} with {active_method_name} ===")
            
            for metric in metrics_to_plot:
                try:
                    plot_metric_results(
                        metric_name=metric,
                        F1=F1, F2=F2, F3=F3,
                        hm=HM,
                        num_dm_dec=num_dm_dec,
                        dataset_fold=DATASET_FOLDS[0],
                        sub_fold=sub_fold,
                        save_figs=True,
                        show_figs=False
                    )
                except Exception as e:
                    print(f"Failed to plot {metric} for {sub_fold}: {e}")
            
            # Diagnostic plotting for BALD+US is now handled internally in simulation.py
            # controlled by the PLOT_MAPE_FIT flag.
