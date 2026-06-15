"""
Net-benefit stopping criterion:
    NB(alpha) = ( r(alpha) - r_0 )  -  lambda * rho(alpha),
the ASRS quality *acquired* through elicitation minus lambda times the budget spent to
acquire it. Both terms are in [0,1]; lambda is the exchange rate (how many units of
retained quality one unit of budget is worth). Unlike a ratio, NB is concave-ish in alpha
(the gain saturates while the budget keeps growing), so it has a well-defined INTERIOR
maximum: the point past which an extra query costs more budget than the quality it adds.

Uses the cached KL signals (scratch/stopping_raw.pkl); no recompute.
"""
import os, pickle, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stopping_eval import REGIMES, ALGOS, RAW
from plot_eta_gain import curves, ALPHAS

PANEL = {('FTRL', 'US'): (0, 0), ('FTRL', 'BALD'): (0, 1),
         ('BAYES', 'US'): (1, 0), ('BAYES', 'BALD'): (1, 1)}
RCOL = {'exp_dataset': 'tab:blue', 'exp_dataset_inc': 'tab:red'}
RLAB = {'exp_dataset': r'$0\%$ inconsistency', 'exp_dataset_inc': r'$20\%$ inconsistency'}


def main(lam):
    raw = pickle.load(open(RAW, 'rb'))
    A = ALPHAS[ALPHAS >= 0.01]                       # start the axis at 0.01
    plt.rcParams.update({'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 16,
                         'legend.fontsize': 13})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    print(f'% argmax NB  (lambda={lam}):')
    for engine, method, _, _ in ALGOS:
        ax = axes[PANEL[(engine, method)]]
        for ds, _ in REGIMES:
            bud, ret, gain = curves(raw[(ds, engine, method)], A)
            nb = gain - lam * bud
            c = RCOL[ds]
            ax.plot(A, nb, color=c, lw=2.4, label=RLAB[ds])
            k = int(np.argmax(nb))
            ax.plot(A[k], nb[k], 'o', color=c, ms=10, mec='k', mew=1, zorder=5)
            print(f'%   {ds:16} {engine}-{method:5}: alpha={A[k]:.3f}  '
                  f'NB={nb[k]:.3f}  ret={ret[k]*100:.0f}%  bud={bud[k]*100:.0f}%')
        ax.axhline(0.0, ls=':', color='gray')
        ax.set_xscale('log'); ax.set_xlim(0.01, 0.95)
        ax.set_title(f'{engine}-{method}')
        ax.grid(alpha=0.3, which='both')
        if PANEL[(engine, method)][1] == 0:
            ax.set_ylabel('Net Benefit (NB)')
        if PANEL[(engine, method)][0] == 1:
            ax.set_xlabel(r'Threshold ($\alpha$)')
    axes[0][0].legend()
    fig.tight_layout()
    out = os.path.join('scratch', 'stopping_nb_alpha.png')
    fig.savefig(out, dpi=140); print('wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam', type=float, default=1.0)
    args = ap.parse_args()
    main(args.lam)