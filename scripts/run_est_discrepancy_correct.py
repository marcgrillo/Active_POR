"""
Score-level estimation-error gap under CORRECT links:
  FTRL-LIN on lin_dataset_pwl (correct, epsilon^Lap>1)
  FTRL-BT  on exp_dataset     (correct, epsilon^Lap<1, control)
Compares the Laplace+Taylor acquisition score to a gold-standard IS-MC reference on the
SAME posterior/link -> isolates approximation error from link error AND learnability.
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import measure_est_discrepancy as M

ALG_DS = {'FTRL-LIN': 'lin_dataset_pwl', 'FTRL-BT': 'exp_dataset'}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--f1', type=int, default=30)
    ap.add_argument('--f2', type=int, nargs='+', default=[2, 6, 10])
    ap.add_argument('--hms', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--steps', type=int, nargs='+', default=[1, 3, 5, 8, 11, 14, 18, 22, 27, 33])
    ap.add_argument('--nsamp', type=int, default=40000)
    a = ap.parse_args()

    allrows = []
    for alg, ds in ALG_DS.items():
        M.DS = ds                      # point each link at the dataset its own link generates
        for f2 in a.f2:
            allrows += M.run(alg, a.f1, f2, a.hms, a.steps, a.nsamp)
    df = pd.DataFrame(allrows)
    if df.empty:
        print('No usable points (ESS too low?).'); raise SystemExit
    df.to_csv(os.path.join('scratch', 'est_discrepancy_correct.csv'), index=False)
    pd.set_option('display.float_format', lambda v: f'{v:.3f}')
    df['eps_gt1'] = df['epsilon'] > 1
    print(f'\nUsable points: {len(df)} (ESS>=2%)')
    print('\n=== Mean by alg x epsilon-regime (correct links) ===')
    print(df.groupby(['alg', 'eps_gt1']).agg(
        n=('hm', 'count'), epsilon=('epsilon', 'mean'), ess=('ess', 'mean'),
        bald_disc=('bald_disc', 'mean'), us_disc=('us_disc', 'mean'),
        bald_top1_miss=('bald_top1_miss', 'mean')).to_string())
    print('\n=== Spearman epsilon^Lap vs score discrepancy ===')
    for col in ['bald_disc', 'us_disc', 'bald_top1_miss']:
        rho, p = spearmanr(df.epsilon, df[col])
        print(f'  {col:16}: rho={rho:+.3f} (p={p:.1e})')
