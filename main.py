import numpy as np
from experiments.runner import run_batch_experiments
from experiments.metrics import BenchmarkRunner
from common.utils import parse_subfold_string

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Experiment Parameters
F1 = [10]       # Alternatives
F2 = [2]        # Criteria
F3 = [100]       # % of pairwise comparisons
DATASET_FOLDS = ['datasets']

# Algorithms to Benchmark
TARGET_METHODS = [
    #'BAYES_LIN_BALD',
    'BAYES_BT_BALD',
    #'FTRL_LIN_BALD',
    #'FTRL_BT_BALD',
    #'BAYES_LIN_US',
    #'BAYES_BT_US',
    #'FTRL_LIN_US',
    #'FTRL_BT_US',
    #'FTRL_LIN_BALD+US',
    #'BAYES_LIN_BALD+US',
] 

# Shared Parameters
HM_0 = 200 # Number of Human Models to use for BOTH simulation and metrics
CALCULATE_METRICS = True

# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Run Experiments
    for sub_fold in TARGET_METHODS:
        alg_name, active_method_name = parse_subfold_string(sub_fold)
        print(f"\n>>> Running: {alg_name} with {active_method_name}")

        algo_type, model_type = alg_name.split('-')
        if algo_type == 'BAYES': HM = int(HM_0/10)
        else: HM = HM_0
        
        run_batch_experiments(
            F1, F2, F3, 
            sub_fold=sub_fold, 
            dataset_folds=DATASET_FOLDS, 
            alg=alg_name, 
            active_method=active_method_name, 
            overwrite=False,
            hm=HM  # Pass the limit here
        )
        
    # 2. Calculate Metrics
    if CALCULATE_METRICS:
        # Note: Using first F1/F2/F3 config for metric calculation setup
        f1, f2, f3 = F1[0], F2[0], F3[0]
        num_dm_dec = int(np.round(f3 * (f1 * (f1 - 1) / 200)))
        
        for sub_fold in TARGET_METHODS:
            alg_name, active_method_name = parse_subfold_string(sub_fold)

            algo_type, model_type = alg_name.split('-')
            if alg_name == 'BAYES': HM = int(HM/10)

            print(f"\n=== Calculating Metrics for {alg_name} with {active_method_name} ===")
            runner = BenchmarkRunner(
                dataset_fold=DATASET_FOLDS[0],
                sub_fold=sub_fold, # Metrics for first method in list
                num_subint=3,
                hm=HM, # Use same limit here
                F1=F1, F2=F2, F3=F3,
                num_dm_dec=num_dm_dec
            )
            runner.compute_perc_inc(force=CALCULATE_METRICS)
            runner.compute_metrics("poi", force=CALCULATE_METRICS)
            runner.compute_metrics("rai", force=CALCULATE_METRICS)
            runner.compute_asrs(force=CALCULATE_METRICS)
            runner.compute_aios(force=CALCULATE_METRICS)
            runner.compute_asps(force=CALCULATE_METRICS)