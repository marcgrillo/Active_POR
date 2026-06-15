"""
FTRL-BT BALD+US with diagnostics ON on exp_dataset, to obtain epsilon^Lap (the
relative cubic/quadratic Laplace ratio) paired with ASRS, matching the FTRL-LIN
diagnostics already collected on exp_dataset. This lets us correlate epsilon^Lap
with FTRL BALD performance across links (BT: eps<1, BALD works; LIN: eps>1, fails).

overwrite=True is required so the simulation re-runs each step and logs epsilon
(skipped steps log nothing). The FTRL-BT trajectories are statistically equivalent
to the H1 runs; only error_scores.csv is added.
"""
from experiments.runner import run_batch_experiments

F1 = [10, 30]
F2 = [2, 6, 10]
F3 = [100]
DS = 'exp_dataset'
METHODS = ['BALD', 'US']
HM = 10

if __name__ == '__main__':
    for method in METHODS:
        sf = f'FTRL-BT_{method}'
        print(f'\n=== {DS} | {sf} | diagnostics ON ===')
        run_batch_experiments(
            F1=F1, F2=F2, F3=F3,
            sub_fold=sf, dataset_folds=[DS],
            alg='FTRL-BT', active_method=method,
            overwrite=True, hm=HM,
            n_samples_mc=2000, use_is_bald=True,
            use_mh_sampler=True, use_hmc_sampler=True,
            num_cores=8, n_samples_mcmc=2000,
            compute_error_diagnostics=True,
            check_passive_algs_completed=False,
            adaptive_mcmc=False,
        )