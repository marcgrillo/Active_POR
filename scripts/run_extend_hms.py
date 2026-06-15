"""
Extend all 8 BAYES conditions to the per-dataset HM ceiling, to power up the
link-misspecification analysis. overwrite=False -> only the new HMs are computed.

  exp_dataset_inc (logit 20%): -> 25 HMs   (BAYES-LIN/BT x US/BALD)
  lin_dataset_pwl (linear)    : -> 20 HMs

Usage:  python run_extend_hms.py --cores 10
"""
import argparse
from experiments.runner import run_batch_experiments

F2 = [2, 6, 10]
F3 = [100]
F1 = [30]
JOBS = [('exp_dataset_inc', 25), ('lin_dataset_pwl', 20)]
MODELS = ['BAYES-LIN', 'BAYES-BT']
METHODS = ['US', 'BALD']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cores', type=int, default=10)
    args = p.parse_args()
    for ds, hm in JOBS:
        for alg in MODELS:
            for method in METHODS:
                sf = f'{alg}_{method}'
                print(f'\n=== {ds} | {sf} | -> {hm} HMs ===')
                run_batch_experiments(
                    F1=F1, F2=F2, F3=F3,
                    sub_fold=sf, dataset_folds=[ds],
                    alg=alg, active_method=method,
                    overwrite=False, hm=hm,
                    n_samples_mc=2000, use_is_bald=True,
                    use_mh_sampler=True, use_hmc_sampler=True,
                    num_cores=args.cores, n_samples_mcmc=2000,
                    compute_error_diagnostics=False,
                    check_passive_algs_completed=False,
                    adaptive_mcmc=False,
                    max_steps=60,          # only the early horizon (analysis uses t<=50)
                )


if __name__ == '__main__':
    main()
