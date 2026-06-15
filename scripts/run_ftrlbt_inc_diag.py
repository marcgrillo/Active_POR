"""
FTRL-BT (US + BALD) on exp_dataset_inc (logit, 20% inconsistency) with diagnostics ON,
to obtain epsilon^Lap under a CORRECTLY-SPECIFIED BT link at a non-zero noise level.
This matches the est-error study design: each link is evaluated on the data its own link
generates (BT<->logit, LIN<->linear), so any FTRL failure isolates to the Laplace
approximation, not link misspecification.

Written to a SEPARATE sub_fold ('..._estdiag') so the existing full-horizon FTRL-BT
H1 trajectories on exp_dataset_inc (used by the performance comparison) are NOT clobbered.
Capped at 60 steps (analysis horizon t<=50).
"""
from experiments.runner import run_batch_experiments

F1 = [30]
F2 = [2, 6, 10]
F3 = [100]
DS = 'exp_dataset_inc'
METHODS = ['BALD', 'US']
HM = 10
MAX_STEPS = 60

if __name__ == '__main__':
    for method in METHODS:
        sf = f'FTRL-BT_{method}_estdiag'
        print(f'\n=== {DS} | {sf} | diagnostics ON | max_steps={MAX_STEPS} ===')
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
            max_steps=MAX_STEPS,
        )
    print('\nDONE FTRL-BT estdiag runs.')
