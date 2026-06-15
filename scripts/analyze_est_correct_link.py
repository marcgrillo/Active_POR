"""
Estimation-error (Laplace) study under CORRECTLY-SPECIFIED links only
(BT<->logit data, LIN<->linear data), so any FTRL failure isolates to the Laplace
approximation rather than link misspecification.

GATE  : epsilon^Lap per correct-link FTRL arm (mean +-2SEM, % steps >1, early vs late).
GAP   : FTRL active-over-random ASRS gain per arm (mean +-2SEM, Wilcoxon p), and the
        within-arm Spearman rho(epsilon, gain).
CONTROL: BAYES-LIN (MCMC, no Laplace) active-over-random gain on lin_dataset_pwl,
        proving active learning IS possible there with the correct link -> FTRL-LIN's
        failure is the Laplace approximation, not the data being too hard.
"""
import os, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from experiments.aggregate import load_curve

F3 = 100
F2 = [2, 6, 10]
SAMP = 'samples'

# (label, dataset, FTRL arm sub_fold, generative link match)
CORRECT = [
    ('BT  / logit 0%',  'exp_dataset',     'FTRL-BT_US'),
    ('BT  / logit 0%',  'exp_dataset',     'FTRL-BT_BALD'),
    ('BT  / logit 20%', 'exp_dataset_inc', 'FTRL-BT_US_estdiag'),
    ('BT  / logit 20%', 'exp_dataset_inc', 'FTRL-BT_BALD_estdiag'),
    ('LIN / linear 42%','lin_dataset_pwl', 'FTRL-LIN_US'),
    ('LIN / linear 42%','lin_dataset_pwl', 'FTRL-LIN_BALD'),
]


def eps_per_table(ds, arm, f1, f2, tmax):
    base = os.path.join(SAMP, ds, f'f1_{f1}_f2_{f2}_f3_{F3}', arm)
    out = {}
    for d in sorted(glob.glob(os.path.join(base, 'table_*'))):
        h = int(os.path.basename(d).split('_')[1])
        p = os.path.join(d, 'error_scores.csv')
        if not os.path.exists(p):
            continue
        e = pd.read_csv(p)['epsilon_t'].replace([np.inf, -np.inf], np.nan).to_numpy()[:tmax]
        out[h] = e
    return out


def gain_per_hm(ds, arm, f1, f2, tmax):
    ndm = int(round(F3 * f1 * (f1 - 1) / 200))
    act = load_curve('asrs', ds, arm, f1, f2, F3, ndm, 'active')
    pas = load_curve('asrs', ds, arm, f1, f2, F3, ndm, 'passive')
    if not act.size or not pas.size:
        return None
    T = min(tmax, act.shape[0], pas.shape[0])
    return np.nanmean(act[:T], axis=0) - np.nanmean(pas[:T], axis=0)


def collect(ds, arm, f1, tmax):
    eps_cell, gain_cell = [], []   # per (f2,HM) cell: mean epsilon, gain
    for f2 in F2:
        ep = eps_per_table(ds, arm, f1, f2, tmax)
        g = gain_per_hm(ds, arm, f1, f2, tmax)
        if g is None or not ep:
            continue
        for h, ev in ep.items():
            me = np.nanmean(ev)
            if h < len(g) and np.isfinite(me) and np.isfinite(g[h]):
                eps_cell.append(me); gain_cell.append(g[h])
    return np.array(eps_cell), np.array(gain_cell)


def frac_hi(ds, arm, f1, tmax):
    vals = []
    for f2 in F2:
        for h, ev in eps_per_table(ds, arm, f1, f2, tmax).items():
            vals.extend(ev[np.isfinite(ev)].tolist())
    vals = np.array(vals)
    return (vals > 1).mean() if vals.size else np.nan, vals.mean() if vals.size else np.nan


def s2(v):
    v = np.array([x for x in v if np.isfinite(x)])
    return v.mean(), 2 * v.std(ddof=1) / np.sqrt(len(v)), wilcoxon(v).pvalue, len(v)


def main():
    f1, tmax = 30, 50
    print(f'================ CORRECT-LINK est-error  (f1={f1}, t<={tmax}) ================')
    print(f'{"arm":34} {"eps mean":>9} {">1":>5} {"gain":>22} {"rho(eps,gain)":>16}')
    for lab, ds, arm in CORRECT:
        eps, gain = collect(ds, arm, f1, tmax)
        if len(eps) < 4:
            print(f'{lab+" "+arm:34} -- no data (n={len(eps)}) [run pending?]')
            continue
        fh, em = frac_hi(ds, arm, f1, tmax)
        gm, gse, gp, n = s2(gain)
        r, rp = spearmanr(eps, gain)
        print(f'{lab+" / "+arm.split("_")[-1]:34} {em:9.2f} {fh*100:4.0f}% '
              f'{gm:+.4f}+-{gse:.4f} p={gp:.0e}  rho={r:+.2f} p={rp:.2f} (n={n})')

    print('\n---------------- CONTROL: BAYES-LIN (no Laplace) on lin_dataset_pwl ----------------')
    for arm in ['BAYES-LIN_US', 'BAYES-LIN_BALD']:
        g = []
        for f2 in F2:
            gg = gain_per_hm('lin_dataset_pwl', arm, f1, f2, tmax)
            if gg is not None:
                g.extend(gg.tolist())
        gm, gse, gp, n = s2(g)
        print(f'  {arm:16}: active-over-random gain={gm:+.4f}+-{gse:.4f}  p={gp:.1e} (n={n})')


if __name__ == '__main__':
    main()
