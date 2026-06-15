"""
Full experiment run: all strategies across F1={10,30,50} x F2={2,6,10},
consistent and 20% inconsistent datasets.

Usage:
    python run_experiments.py --mode ftrl        # FTRL only (~1h, 10 cores)
    python run_experiments.py --mode bayes       # BAYES only (overnight, 16 cores)
    python run_experiments.py --mode metrics     # compute RAI + ASRS after runs finish
    python run_experiments.py --mode aggregate   # print comparison tables
    python run_experiments.py --mode all         # everything in sequence
"""

import argparse
import numpy as np
from experiments.runner import run_batch_experiments, adaptive_n_samples
from experiments.metrics import BenchmarkRunner

F1 = [10, 30, 50]
F2 = [2, 6, 10]
F3 = [100]
DATASETS = {'consistent': 'exp_dataset', 'inconsistent': 'exp_dataset_inc'}

# f1=50 excluded from BAYES: 10000 adaptive MCMC samples × 1225 steps × 20 HMs is infeasible
F1_BAYES = [10, 30]

FTRL_STRATEGIES = ['BALD', 'US', 'POLY', 'CHEB', 'MAXREGRET']
BAYES_STRATEGIES = ['BALD', 'US', 'RANKUNC', 'POLY', 'CHEB', 'MAXREGRET']

HM_FTRL  = 10
HM_BAYES = 20
GAMMA    = 0.75
OVERWRITE = False   # set True to re-run completed configs


def run_ftrl(num_cores=10):
    alg = 'FTRL-BT'
    for cond, ds in DATASETS.items():
        for method in FTRL_STRATEGIES:
            sf = f'{alg}_{method}'
            print(f'\n[FTRL] {cond} | {sf}')
            run_batch_experiments(
                F1=F1, F2=F2, F3=F3,
                sub_fold=sf, dataset_folds=[ds],
                alg=alg, active_method=method,
                overwrite=OVERWRITE, hm=HM_FTRL,
                n_samples_mc=2000, use_is_bald=True,
                use_mh_sampler=True, use_hmc_sampler=True,
                num_cores=num_cores, n_samples_mcmc=2000,
                compute_error_diagnostics=False,
                check_passive_algs_completed=False,
                adaptive_mcmc=False,   # FTRL doesn't use MCMC
            )


def run_bayes(num_cores=16):
    alg = 'BAYES-BT'
    for cond, ds in DATASETS.items():
        for method in BAYES_STRATEGIES:
            sf = f'{alg}_{method}'
            print(f'\n[BAYES] {cond} | {sf}')
            run_batch_experiments(
                F1=F1_BAYES, F2=F2, F3=F3,
                sub_fold=sf, dataset_folds=[ds],
                alg=alg, active_method=method,
                overwrite=OVERWRITE, hm=HM_BAYES,
                n_samples_mc=2000, use_is_bald=True,
                use_mh_sampler=True, use_hmc_sampler=True,
                num_cores=num_cores, n_samples_mcmc=2000,
                compute_error_diagnostics=False,
                check_passive_algs_completed=False,
                adaptive_mcmc=True,    # scales with f1, f2
            )


def run_metrics():
    ftrl_strats  = [(f'FTRL-BT_{m}',  HM_FTRL,  F1)       for m in FTRL_STRATEGIES]
    bayes_strats = [(f'BAYES-BT_{m}', HM_BAYES, F1_BAYES)  for m in BAYES_STRATEGIES]
    for cond, ds in DATASETS.items():
        for sf, hm, f1_list in ftrl_strats + bayes_strats:
            for f1 in f1_list:
                for f2 in F2:
                    ndm = int(round(100 * f1 * (f1 - 1) / 200))
                    # (inner loop body unchanged, just dedented)
                    print(f'Metrics: {ds} f1={f1} f2={f2} {sf}')
                    try:
                        r = BenchmarkRunner(
                            dataset_fold=ds, sub_fold=sf, num_subint=3,
                            hm=hm, F1=[f1], F2=[f2], F3=F3, num_dm_dec=ndm,
                        )
                        r.compute_metrics('rai', force=False)
                        r.compute_asrs(force=False)
                    except Exception as e:
                        print(f'  FAILED: {e}')


def run_aggregate():
    from experiments.aggregate import build_table_multi, to_latex
    import pandas as pd

    ftrl_configs  = [(f1, f2, 100) for f1 in F1       for f2 in F2]
    bayes_configs = [(f1, f2, 100) for f1 in F1_BAYES for f2 in F2]
    pd.set_option('display.float_format', lambda v: f'{v:.4f}')

    for cond, ds in DATASETS.items():
        print(f'\n{"="*60}')
        print(f'FTRL-BT | {cond.upper()} | {len(ftrl_configs)} configs | gamma={GAMMA}')
        print('='*60)
        strats = [f'FTRL-BT_{m}' for m in FTRL_STRATEGIES]
        df = build_table_multi(ds, strats, ftrl_configs, metric='asrs', gamma=GAMMA)
        print(df.to_string(index=False))
        print(to_latex(df, caption=f'FTRL-BT ASRS — {cond}, $\\gamma$={GAMMA}',
                       label=f'tab:ftrl_{cond}'))

        print(f'\n{"="*60}')
        print(f'BAYES-BT | {cond.upper()} | {len(bayes_configs)} configs | gamma={GAMMA}')
        print('='*60)
        strats = [f'BAYES-BT_{m}' for m in BAYES_STRATEGIES]
        df = build_table_multi(ds, strats, bayes_configs, metric='asrs', gamma=GAMMA)
        print(df.to_string(index=False))
        print(to_latex(df, caption=f'BAYES-BT ASRS — {cond}, $\\gamma$={GAMMA}',
                       label=f'tab:bayes_{cond}'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['ftrl', 'bayes', 'metrics', 'aggregate', 'all'],
                   default='all')
    p.add_argument('--ftrl_cores', type=int, default=10)
    p.add_argument('--bayes_cores', type=int, default=16)
    args = p.parse_args()

    if args.mode in ('ftrl', 'all'):
        run_ftrl(num_cores=args.ftrl_cores)
    if args.mode in ('bayes', 'all'):
        run_bayes(num_cores=args.bayes_cores)
    if args.mode in ('metrics', 'all'):
        run_metrics()
    if args.mode in ('aggregate', 'all'):
        run_aggregate()