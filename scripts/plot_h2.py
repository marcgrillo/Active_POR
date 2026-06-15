"""
Aggregate H2 error_scores.csv into (E_est, E_link) points and draw scatter plots,
both per-(config,HM) and per-config(HM-averaged), so we can choose the representation.

E_est  = mean rho_t over steps of a trajectory.
E_link = final relative link error (E_link_rel). Flagged/needs care where rho_t==1
         throughout (link error not meaningful), per the design note.
"""
import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

F2 = [2, 6, 10]
F3 = 100
DATASETS = {'exp_dataset': 'misspec', 'lin_dataset_pwl': 'correct'}
ALGS = ['FTRL-LIN', 'BAYES-LIN']
METHODS = ['BALD', 'US']


def collect(f1_list):
    rows = []
    for ds, dlab in DATASETS.items():
        for f1 in f1_list:
            for f2 in F2:
                cfg = f'f1_{f1}_f2_{f2}_f3_{F3}'
                for alg in ALGS:
                    for method in METHODS:
                        sf = f'{alg}_{method}'
                        base = os.path.join('samples', ds, cfg, sf)
                        for tdir in glob.glob(os.path.join(base, 'table_*')):
                            csv = os.path.join(tdir, 'error_scores.csv')
                            if not os.path.exists(csv):
                                continue
                            try:
                                df = pd.read_csv(csv)
                            except Exception:
                                continue
                            if df.empty or 'rho_t' not in df:
                                continue
                            rho = df['rho_t'].to_numpy(dtype=float)
                            elink = df['E_link_rel'].to_numpy(dtype=float)
                            hm = int(os.path.basename(tdir).split('_')[1])
                            rows.append(dict(
                                dataset=dlab, alg=alg.split('-')[0], method=method,
                                f1=f1, f2=f2, hm=hm,
                                E_est=float(np.nanmean(rho)),
                                frac_rho1=float(np.mean(rho >= 1.0)),
                                E_link=float(elink[-1]) if len(elink) else np.nan,
                            ))
    return pd.DataFrame(rows)


def scatter(df, fname, by_config=False):
    if by_config:
        df = (df.groupby(['dataset', 'alg', 'method', 'f1', 'f2'], as_index=False)
                .agg(E_est=('E_est', 'mean'), E_link=('E_link', 'mean'),
                     frac_rho1=('frac_rho1', 'mean')))
    fig, ax = plt.subplots(figsize=(7, 6))
    style = {'FTRL': 'o', 'BAYES': 's'}
    color = {'misspec': 'tab:red', 'correct': 'tab:blue'}
    for (alg, dlab), g in df.groupby(['alg', 'dataset']):
        meaningful = g['frac_rho1'] < 1.0
        ax.scatter(g.loc[meaningful, 'E_est'], g.loc[meaningful, 'E_link'],
                   marker=style[alg], c=color[dlab], alpha=0.7, s=60,
                   label=f'{alg} / {dlab}')
        # rho==1 points (E_link not meaningful) as hollow x
        ax.scatter(g.loc[~meaningful, 'E_est'], g.loc[~meaningful, 'E_link'],
                   marker='x', c=color[dlab], alpha=0.3, s=40)
    ax.set_xlabel(r'$\mathcal{E}_{\rm est}$ (mean $\rho_t$)')
    ax.set_ylabel(r'$\mathcal{E}_{\rm link}$ (final, relative)')
    ax.set_title('H2 error scatter — LIN model' + (' (per config)' if by_config else ' (per HM)'))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print('wrote', fname)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--f1', type=int, nargs='+', default=[10])
    args = p.parse_args()
    df = collect(args.f1)
    if df.empty:
        print('No error_scores found yet.')
        raise SystemExit
    pd.set_option('display.width', 170)
    pd.set_option('display.float_format', lambda v: f'{v:.3f}')
    print('\n=== Per (dataset, alg, method) means ===')
    print(df.groupby(['dataset', 'alg', 'method'])
            .agg(E_est=('E_est', 'mean'), E_link=('E_link', 'mean'),
                 frac_rho1=('frac_rho1', 'mean'), n=('hm', 'count'))
            .to_string())
    tag = '_'.join(str(x) for x in args.f1)
    scatter(df, os.path.join('scratch', f'h2_scatter_perHM_f1{tag}.png'), by_config=False)
    scatter(df, os.path.join('scratch', f'h2_scatter_perCfg_f1{tag}.png'), by_config=True)