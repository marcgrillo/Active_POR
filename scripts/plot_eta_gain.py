"""
Gain-based elicitation efficiency:
    zeta(alpha) = ( ASRS_retained(T_stop) - ASRS_retained(start) ) / budget_used,
i.e. the ASRS *improvement* achieved by elicitation per unit of budget. Unlike
eta = retained/budget (which diverges as budget->0 because the baseline retention is
nonzero), the numerator here vanishes when nothing is elicited, so zeta need not be
maximized at budget zero.

Uses the cached KL signals (scratch/stopping_raw.pkl); no recompute.
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stopping_eval import REGIMES, ALGOS, RAW, W_MA

ALPHAS = np.unique(np.round(np.logspace(np.log10(0.001), np.log10(0.95), 45), 4))
PANEL = {('FTRL', 'US'): (0, 0), ('FTRL', 'BALD'): (0, 1),
         ('BAYES', 'US'): (1, 0), ('BAYES', 'BALD'): (1, 1)}
RCOL = {'exp_dataset': 'tab:blue', 'exp_dataset_inc': 'tab:red'}
RLAB = {'exp_dataset': r'$0\%$ inconsistency', 'exp_dataset_inc': r'$20\%$ inconsistency'}


def curves(data, alphas):
    """Per-alpha mean budget, mean retained r, mean gain (r - r_start)."""
    B, R, G = [], [], []
    for a in alphas:
        bs, rs, gs = [], [], []
        for score, asrs in data:
            s = np.array(score, float)
            if not np.isfinite(s).any():
                continue
            ma = np.convolve(np.nan_to_num(s, nan=np.nanmean(s)), np.ones(W_MA) / W_MA, 'same')
            i0 = int(np.where(np.isfinite(s))[0][0])
            s0 = ma[i0]
            if not np.isfinite(s0) or s0 <= 0:
                continue
            T = len(asrs)
            final = np.nanmean(asrs[-max(1, T // 20):])
            if not np.isfinite(final) or final <= 0:
                continue
            hit = np.where(ma < a * s0)[0]; hit = hit[hit >= i0]
            stop = hit[0] if len(hit) else T - 1
            r_start = asrs[i0] / final
            bs.append((stop + 1) / T)
            rs.append(np.clip(asrs[stop] / final, 0, 1.5))
            gs.append(np.clip(asrs[stop] / final, 0, 1.5) - r_start)
        if bs:
            B.append(np.mean(bs)); R.append(np.mean(rs)); G.append(np.mean(gs))
    return np.array(B), np.array(R), np.array(G)


def main():
    raw = pickle.load(open(RAW, 'rb'))
    plt.rcParams.update({'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 16,
                         'legend.fontsize': 13})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for engine, method, _, _ in ALGOS:
        ax = axes[PANEL[(engine, method)]]
        for ds, _ in REGIMES:
            bud, ret, gain = curves(raw[(ds, engine, method)], ALPHAS)
            zeta = gain / np.clip(bud, 1e-6, None)
            c = RCOL[ds]
            ax.plot(ALPHAS, zeta, color=c, lw=2.4, label=RLAB[ds])
            k = int(np.argmax(zeta))
            ax.plot(ALPHAS[k], zeta[k], 'o', color=c, ms=10, mec='k', mew=1, zorder=5)
        ax.set_xscale('log')
        ax.set_title(f'{engine}-{method}')
        ax.grid(alpha=0.3, which='both')
        if PANEL[(engine, method)][1] == 0:
            ax.set_ylabel(r'$\zeta=$ (ASRS gain) / budget')
        if PANEL[(engine, method)][0] == 1:
            ax.set_xlabel(r'threshold $\alpha$')
    axes[0][0].legend()
    fig.suptitle(r'Gain-based efficiency $\zeta(\alpha)=(r(T_{\rm stop})-r_0)/\rho$'
                 r'  (dot = maximum)', y=0.995, fontsize=18)
    fig.tight_layout()
    out = os.path.join('scratch', 'stopping_zeta_alpha.png')
    fig.savefig(out, dpi=140); print('wrote', out)

    # report argmax per arm/regime
    print('\n% argmax zeta (gain/budget):')
    for engine, method, _, _ in ALGOS:
        for ds, _ in REGIMES:
            bud, ret, gain = curves(raw[(ds, engine, method)], ALPHAS)
            zeta = gain / np.clip(bud, 1e-6, None); k = int(np.argmax(zeta))
            print(f'%   {ds:16} {engine}-{method:5}: alpha={ALPHAS[k]:.3f}  '
                  f'zeta={zeta[k]:.2f}  ret={ret[k]*100:.0f}%  bud={bud[k]*100:.0f}%')


if __name__ == '__main__':
    main()