"""
Sensitivity (FTRL-BT, consistent regime only): Delta AULC = AULC(method) - AULC(Random)
vs F1 and vs F2, as a GROUPED BAR plot (three adjacent bars -- US, BALD, best baseline --
per factor value) with error bars (2 x SEM of the paired gain).
Layout: 2 rows (ASRS, ASPS) x 2 cols (vs F1, vs F2).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.aggregate import load_curve, aulc_per_hm

F3 = 100
ENGINE = 'FTRL-BT'
DATASET = 'exp_dataset'          # consistent only
METRICS = [('asrs', 'ASRS'), ('asps', 'ASPS')]
F1_ALL = [10, 30]
F2_ALL = [2, 6, 10]
METHODS = ['US', 'BALD']
BASELINES = ['POLY', 'CHEB', 'MAXREGRET']
COL = {'US': 'tab:orange', 'BALD': 'tab:red'}

plt.rcParams.update({
    'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
})


def aulc_vals(metric, sub_fold, f1, f2, track):
    ndm = int(round(F3 * f1 * (f1 - 1) / 200))
    c = load_curve(metric, DATASET, sub_fold, f1, f2, F3, ndm, track)
    return aulc_per_hm(c) if c.size else np.array([])


def dAULC_stats(metric, sub_fold, fix_factor, fix_val):
    """Paired gain mean and 2*SEM, pooling the other factor + HMs."""
    pairs = [(fix_val, f2) for f2 in F2_ALL] if fix_factor == 'F1' \
        else [(f1, fix_val) for f1 in F1_ALL]
    gains = []
    for f1, f2 in pairs:
        m = aulc_vals(metric, sub_fold, f1, f2, 'active')
        r = aulc_vals(metric, f'{ENGINE}_BALD', f1, f2, 'passive')
        n = min(len(m), len(r))
        if n:
            gains += (m[:n] - r[:n]).tolist()
    g = np.array([x for x in gains if np.isfinite(x)])
    if not g.size:
        return np.nan, 0.0
    return g.mean(), 2 * g.std(ddof=1) / np.sqrt(len(g))


def best_baseline(metric):
    best, score = None, -np.inf
    for b in BASELINES:
        vals = []
        for f1 in F1_ALL:
            for f2 in F2_ALL:
                vals += aulc_vals(metric, f'{ENGINE}_{b}', f1, f2, 'active').tolist()
        if vals and np.nanmean(vals) > score:
            best, score = b, np.nanmean(vals)
    return best


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    width = 0.26
    for ri, (metric, mlab) in enumerate(METRICS):
        bb = best_baseline(metric)
        series = METHODS + [bb]
        for ci, (factor, vals) in enumerate([('F1', F1_ALL), ('F2', F2_ALL)]):
            ax = axes[ri][ci]
            xpos = np.arange(len(vals))
            for k, m in enumerate(series):
                means, errs = [], []
                for v in vals:
                    mu, ci2 = dAULC_stats(metric, f'{ENGINE}_{m}', factor, v)
                    means.append(mu); errs.append(ci2)
                c = COL.get(m, 'tab:green')
                lab = m if m in METHODS else f'Best baseline ({m})'
                ax.bar(xpos + (k - 1) * width, means, width, yerr=errs, capsize=4,
                       color=c, label=lab, edgecolor='white', linewidth=0.6)
            ax.axhline(0, color='gray', lw=0.9)
            ax.set_xticks(xpos)
            ax.set_xticklabels([f'${factor[0]}_{factor[1]}={v}$' for v in vals])
            if ci == 0:
                ax.set_ylabel(rf'{mlab}: $\Delta$AULC (vs Random)')
            ax.grid(alpha=0.3, axis='y')
            if ri == 0 and ci == 0:
                ax.legend()
    fig.suptitle(r'AULC gain sensitivity to $F_1$, $F_2$', y=0.995, fontsize=20)
    fig.tight_layout()
    out = os.path.join('scratch', 'sensitivity_asrs_asps_f1f2.png')
    fig.savefig(out, dpi=140)
    print('wrote', out)


if __name__ == '__main__':
    main()