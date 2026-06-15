"""
Post-hoc link-error E_link for BAYES, computed from saved per-step posteriors + the query
history -- no re-simulation. Replicates the in-simulation diagnostic (simulation.py BAYES
branch): at each step, with the posterior held BEFORE the query, the realised-answer
surprise S_t and the model-expected surprise B_t give the martingale residual
Delta_t = S_t - B_t; E_link_rel = |sum Delta| / sum B.

Covers the cells missing error_scores.csv:
  exp_dataset_inc (logit 20%): BAYES-LIN/BT x US/BALD
  lin_dataset_pwl (linear)   : BAYES-LIN/BT x US/BALD
"""
import os, argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from common import utils
from mcda.models import PiecewiseLinearTransformer

F3 = 100
F2 = [2, 6, 10]
DATASETS = {'exp_dataset_inc': 'logit-20%', 'lin_dataset_pwl': 'linear'}
SPEC = {('LIN', 'exp_dataset_inc'): 'misspec', ('LIN', 'lin_dataset_pwl'): 'correct',
        ('BT', 'exp_dataset_inc'): 'correct', ('BT', 'lin_dataset_pwl'): 'misspec'}


def link(model, u):
    p = 0.5 * (1.0 + u) if model == 'LIN' else expit(u)
    return np.clip(p, 1e-9, 1.0 - 1e-9)


def run_elink(Xf, base, prefs, ndm, model, max_t=50):
    """Cumulative E_link_rel over the first max_t steps (winner-first prefs -> answer=1),
    so HMs with different trajectory lengths are compared over a consistent horizon."""
    sum_D = sum_B = 0.0
    for j in range(2, min(ndm, max_t) + 1):
        wp = os.path.join(base, f'{j-1}_active.npy')      # posterior before query j
        if not os.path.exists(wp) or j - 1 > len(prefs):
            continue
        samples = np.atleast_2d(np.load(wp))
        win, los = int(prefs[j - 1][0]), int(prefs[j - 1][1])   # winner, loser
        vd = Xf[win] - Xf[los]
        u = vd @ samples.T
        p1 = link(model, u); p0 = 1.0 - p1
        pt1, pt0 = p1.mean(), p0.mean()
        am1 = p1 / p1.sum(); am0 = p0 / p0.sum()
        St1 = np.sum(am1 * np.log(p1 / pt1))
        St0 = np.sum(am0 * np.log(p0 / pt0))
        Bt = pt1 * St1 + pt0 * St0
        sum_D += (St1 - Bt)            # realised answer = winner wins = answer 1
        sum_B += Bt
    return abs(sum_D) / (sum_B + 1e-6) if sum_B > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--f1', type=int, default=30)
    ap.add_argument('--hm', type=int, default=10); args = ap.parse_args()
    f1 = args.f1; ndm = int(round(F3 * f1 * (f1 - 1) / 200))
    rows = []
    for ds, dlab in DATASETS.items():
        for model in ['LIN', 'BT']:
            for method in ['US', 'BALD']:
                sf = f'BAYES-{model}_{method}'
                for f2 in F2:
                    try:
                        tables, _, _, _ = utils.read_dataset(os.path.join('datasets', ds), f1, f2, F3)
                    except FileNotFoundError:
                        continue
                    cfg = f'f1_{f1}_f2_{f2}_f3_{F3}'
                    for hm in range(args.hm):
                        base = os.path.join('samples', ds, cfg, sf, f'table_{hm}')
                        pp = os.path.join(base, 'active_prefs.npy')
                        if not os.path.exists(pp):
                            continue
                        prefs = np.load(pp)
                        Xf = PiecewiseLinearTransformer.from_equal_intervals(tables[hm], 3).transform(tables[hm])
                        el = run_elink(Xf, base, prefs, ndm, model)
                        rows.append(dict(dataset=dlab, model=model, method=method, f2=f2,
                                         hm=hm, spec=SPEC[(model, ds)], E_link=el))
    df = pd.DataFrame(rows)
    df.to_csv('scratch/elink_posthoc.csv', index=False)
    print('wrote scratch/elink_posthoc.csv  (n=%d)' % len(df))
    print('\n=== mean E_link by model/method/spec ===')
    print(df.groupby(['model', 'method', 'spec']).E_link.agg(['mean', 'median', 'count']).to_string())


if __name__ == '__main__':
    main()
