"""
Block 4: stopping-criteria evaluation (post-hoc, no re-simulation).

For FTRL-BT and BAYES-BT, US and BALD, consistent regime, we reconstruct two
per-step signals from the saved samples + selected-pair history:

  Rule 1 (acquisition score): the score of the SELECTED query at each step
          US  -> predictive entropy H_t(q_t)
          BALD-> mutual information B_t(q_t)
          stop when its moving average (window W) falls below a threshold.
  Rule 2 (confidence C_t): C_t = mean_{i<j} max(p_ij, 1-p_ij) from the POI matrix;
          stop when C_t exceeds a threshold.

Sweeping each threshold traces a Pareto curve of (fraction of budget used) vs
(fraction of final ASRS retained). A table is emitted at one operating point
(retain >= 95% of final ASRS).
"""
import os, argparse, json, pickle
import numpy as np
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import utils
from mcda.models import PiecewiseLinearTransformer
from inference.engine import PreferenceSampler
from experiments.aggregate import load_curve

DS = 'exp_dataset'
F3 = 100
W_MA = 5  # moving-average window for rule 1


def hbin(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def Ct_value(Xf, w, engine, f1):
    sam = np.atleast_2d(w) if engine == 'FTRL' else w
    vals = Xf @ sam.T                      # (f1, S)
    geq = (vals[:, None, :] >= vals[None, :, :]).mean(axis=2)   # poi[i,j]=P(U_i>=U_j)
    iu = np.triu_indices(f1, k=1)
    return float(np.mean(np.maximum(geq[iu], geq.T[iu])))


def gaussian_kl(mu1, S1, mu0, S0, eps=1e-8):
    """KL( N(mu1,S1) || N(mu0,S0) ) for the 'reference' posterior being the earlier one,
    i.e. how far the updated posterior (t+1) moved from the previous one (t)."""
    d = len(np.ravel(mu0))
    S0 = np.atleast_2d(S0) + eps * np.eye(d)
    S1 = np.atleast_2d(S1) + eps * np.eye(d)
    dmu = np.ravel(mu0) - np.ravel(mu1)
    try:
        S0inv = np.linalg.inv(S0)
        _, ld0 = np.linalg.slogdet(S0)
        _, ld1 = np.linalg.slogdet(S1)
        val = 0.5 * (np.trace(S0inv @ S1) + dmu @ S0inv @ dmu - d + (ld0 - ld1))
        return float(max(val, 0.0))
    except np.linalg.LinAlgError:
        return np.nan


def posterior_moments(Xf, base, prefs, t, engine, alg):
    """Gaussian (mu, Sigma) of the posterior after t observations. FTRL: MAP point +
    Laplace covariance; BAYES: sample mean + sample covariance of the MCMC cloud."""
    p = os.path.join(base, f'{t}_active.npy')
    if not os.path.exists(p):
        return None, None
    w = np.load(p)
    if engine == 'BAYES':
        W = np.atleast_2d(w)
        if W.shape[0] < 2:
            return None, None
        return W.mean(axis=0), np.cov(W.T)
    mu = np.ravel(w)                                   # FTRL MAP
    spr = PreferenceSampler(Xf, prefs[:t].tolist(), Xf.shape[1])
    S = spr.compute_laplace_covariance(mu, alg=alg)
    return mu, np.atleast_2d(S)


def curves_for_hm(Xf, base, prefs, ndm, engine, alg):
    """Per-step stopping signal = KL( posterior_{t+1} || posterior_t ): how much the
    belief still moves with one more comparison. Decreases as elicitation converges."""
    mus = [None] * (ndm + 1)
    Sig = [None] * (ndm + 1)
    for t in range(1, ndm + 1):
        mus[t], Sig[t] = posterior_moments(Xf, base, prefs, t, engine, alg)
    kl = np.full(ndm, np.nan)
    for t in range(1, ndm):                            # kl[t-1] is the gain of obs t+1
        if mus[t] is not None and mus[t + 1] is not None:
            kl[t - 1] = gaussian_kl(mus[t + 1], Sig[t + 1], mus[t], Sig[t])
    return kl


def collect(engine, method, configs, hms, ds=DS):
    """Returns list of (kl, asrs) per HM (kl = posterior-KL stopping signal)."""
    out = []
    alg = f'{engine}-BT'
    for f1, f2 in configs:
        ndm = int(round(F3 * f1 * (f1 - 1) / 200))
        asrs_all = load_curve('asrs', ds, f'{alg}_{method}', f1, f2, F3, ndm, 'active')
        if not asrs_all.size:
            continue
        tables, rankings, dm_prefs, _ = utils.read_dataset(os.path.join('datasets', ds), f1, f2, F3)
        for hm in range(min(hms, asrs_all.shape[1])):
            Xf = PiecewiseLinearTransformer.from_equal_intervals(tables[hm], 3).transform(tables[hm])
            base = os.path.join('samples', ds, f'f1_{f1}_f2_{f2}_f3_{F3}', f'{alg}_{method}', f'table_{hm}')
            pp = os.path.join(base, 'active_prefs.npy')
            if not os.path.exists(pp):
                continue
            prefs = np.load(pp)
            kl = curves_for_hm(Xf, base, prefs, ndm, engine, alg)
            out.append((kl, asrs_all[:, hm]))
    return out


def pareto_rule_relative(data, alphas):
    """Relative posterior-KL rule: stop when the smoothed signal D_t = KL(post_{t+1}||
    post_t) first drops below a fraction alpha of its own INITIAL value, i.e.
    MA(D)_t < alpha * D_0. alpha is a fraction of the starting posterior movement,
    comparable across runs/engines."""
    xs, ys, used = [], [], []
    for a in alphas:
        fracs, rets = [], []
        for score, asrs in data:
            s = np.array(score, float)
            if not np.isfinite(s).any():
                continue
            ma = np.convolve(np.nan_to_num(s, nan=np.nanmean(s)), np.ones(W_MA) / W_MA, 'same')
            i0 = int(np.where(np.isfinite(s))[0][0])      # first valid step
            s0 = ma[i0]                                   # initial smoothed score
            if not np.isfinite(s0) or s0 <= 0:
                continue
            T = len(asrs)
            final = np.nanmean(asrs[-max(1, T // 20):])   # final ASRS (avg of last 5%)
            if not np.isfinite(final) or final <= 0:
                continue
            hit = np.where(ma < a * s0)[0]
            hit = hit[hit >= i0]
            stop = hit[0] if len(hit) else T - 1
            fracs.append((stop + 1) / T)
            rets.append(np.clip(asrs[stop] / final, 0, 1.5))
        if fracs:
            xs.append(np.mean(fracs)); ys.append(np.mean(rets)); used.append(a)
    return np.array(xs), np.array(ys), np.array(used)


REGIMES = [('exp_dataset', r'$0\%$ inconsistency'),
           ('exp_dataset_inc', r'$20\%$ inconsistency')]
ALGOS = [('FTRL', 'US', 'tab:orange', '-'), ('FTRL', 'BALD', 'tab:red', '-'),
         ('BAYES', 'US', 'tab:orange', '--'), ('BAYES', 'BALD', 'tab:red', '--')]


def emit_latex_table(grid, alpha_tab):
    """grid[(ds,engine,method)] -> dict alpha->(budget,retained). Stacked LaTeX table:
    columns = algorithms; the two inconsistency regimes are stacked vertically (0% on
    top, 20% below). Each cell is ASRS-retained / budget-used (%)."""
    head_alg = ' & '.join(r'\textbf{%s-%s}' % (e, m) for e, m, _, _ in ALGOS)
    lines = []
    lines.append(r'\begin{table}[t]\centering\small')
    lines.append(r'\caption{Stopping rule operating characteristics versus the threshold '
                 r'$\alpha$ (fraction of the initial posterior KL $D_t=\mathrm{KL}'
                 r'(\pi_{t+1}\|\pi_t)$). Each cell is \emph{ASRS retained} / '
                 r'\emph{budget used}, in \%, pooled over $F_2\in\{2,6,10\}$ and human '
                 r'models (Bradley--Terry link).}')
    lines.append(r'\label{tab:stopping}')
    lines.append(r'\setlength{\tabcolsep}{6pt}')
    lines.append(r'\begin{tabular}{l cccc}')
    lines.append(r'\toprule')
    lines.append(r'$\alpha$ & ' + head_alg + r' \\')
    for ds, rlab in REGIMES:
        lines.append(r'\midrule')
        lines.append(r'\multicolumn{5}{l}{\emph{' + rlab + r'}} \\')
        for a in alpha_tab:
            cells = []
            for e, m, _, _ in ALGOS:
                bud, ret = grid[(ds, e, m)].get(a, (np.nan, np.nan))
                cells.append('--' if np.isnan(ret)
                             else f'{min(ret, 1.0)*100:.0f}/{bud*100:.0f}')
            lines.append(f'{a:g} & ' + ' & '.join(cells) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


RAW = os.path.join('scratch', 'stopping_raw.pkl')   # expensive per-run data (alpha-free)


def collect_all(configs, hms):
    """raw[(ds,engine,method)] = list of (ct,score,asrs) per run. This is the heavy step
    (loads samples + recomputes FTRL-BALD scores); it does NOT depend on alpha, so we
    cache it and re-derive any alpha grid instantly."""
    raw = {}
    for ds, _ in REGIMES:
        for engine, method, _, _ in ALGOS:
            raw[(ds, engine, method)] = collect(engine, method, configs, hms, ds)
    return raw


def curves_from_raw(raw, alphas):
    """Cheap alpha sweep over cached raw data."""
    curves = {ds: {} for ds, _ in REGIMES}
    for (ds, engine, method), data in raw.items():
        if not data or not any(np.isfinite(d[0]).any() for d in data):
            print(f'no data {ds} {engine}-{method}'); continue
        bud, ret, a1 = pareto_rule_relative(data, alphas)
        curves[ds][f'{engine}-{method}'] = {
            'alpha': a1.tolist(), 'budget': bud.tolist(), 'retained': ret.tolist()}
    return curves


def plot_grid(curves, logx=False):
    """2x2: rows = regime (0% top, 20% bottom); cols = [ASRS retained, budget used].
    x-axis is the threshold alpha, INVERTED (large alpha = stop early = on the left).
    logx=True puts alpha on a log scale (resolves the small-alpha behaviour)."""
    plt.rcParams.update({
        'font.size': 16, 'axes.titlesize': 19, 'axes.labelsize': 17,
        'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 15})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ri, (ds, rlab) in enumerate(REGIMES):
        for engine, method, c, ls in ALGOS:
            cu = curves.get(ds, {}).get(f'{engine}-{method}')
            if cu is None:
                continue
            a1 = np.array(cu['alpha'])
            axes[ri][0].plot(a1, cu['retained'], ls, color=c, lw=2.5, marker='o', ms=4,
                             label=f'{engine}-{method}')
            axes[ri][1].plot(a1, cu['budget'], ls, color=c, lw=2.5, marker='o', ms=4,
                             label=f'{engine}-{method}')
        axes[ri][0].axhline(0.95, ls=':', c='gray')
        axes[ri][0].set_ylabel(f'{rlab}\nASRS retained')
        axes[ri][1].set_ylabel('budget used')
        for cc in (0, 1):
            axes[ri][cc].grid(alpha=0.3, which='both')
            if logx:
                axes[ri][cc].set_xscale('log')
                axes[ri][cc].set_xlim(1.0, 0.0008)     # inverted log
            else:
                axes[ri][cc].set_xlim(0.95, 0.0)       # inverted linear
    axes[0][0].set_title('Fraction of final ASRS retained')
    axes[0][1].set_title('Fraction of budget used at stop')
    for cc in (0, 1):
        axes[1][cc].set_xlabel(r'threshold $\alpha$ (fraction of initial posterior KL)')
    axes[0][0].legend(loc='lower right')
    fig.tight_layout()
    out = os.path.join('scratch', 'stopping_alpha_log.png' if logx else 'stopping_alpha.png')
    fig.savefig(out, dpi=140); print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--f1', type=int, nargs='+', default=[30])
    ap.add_argument('--f2', type=int, nargs='+', default=[2, 6, 10])  # pool configs
    ap.add_argument('--hms', type=int, default=20)
    ap.add_argument('--replot', action='store_true',
                    help='reuse cached raw per-run data (scratch/stopping_raw.pkl); skip '
                         'the ~20-min recompute and just re-sweep alpha + re-plot')
    ap.add_argument('--logx', action='store_true', help='log scale on the alpha axis')
    args = ap.parse_args()
    configs = [(f1, f2) for f1 in args.f1 for f2 in args.f2]
    alphas = np.concatenate(([0.001, 0.01], np.arange(0.05, 0.96, 0.10)))  # +0.001 tail
    alpha_tab = [0.001, 0.01, 0.05, 0.25, 0.45, 0.65, 0.85, 0.95]          # subset on grid

    # Heavy step (alpha-independent) is cached; --replot reuses it for instant restyling.
    if args.replot and os.path.exists(RAW):
        with open(RAW, 'rb') as f:
            raw = pickle.load(f)
        print('loaded raw cache from', RAW)
    else:
        raw = collect_all(configs, args.hms)
        with open(RAW, 'wb') as f:
            pickle.dump(raw, f)
        print('wrote', RAW)

    curves = curves_from_raw(raw, alphas)
    with open(os.path.join('scratch', 'stopping_grid.json'), 'w') as f:
        json.dump(curves, f)
    plot_grid(curves, logx=args.logx)

    # table: snap to nearest computed alpha
    def snap(cu, a):
        al = np.array(cu['alpha']); i = int(np.argmin(np.abs(al - a)))
        return cu['budget'][i], cu['retained'][i]
    grid = {}
    for ds, _ in REGIMES:
        for engine, method, _, _ in ALGOS:
            cu = curves.get(ds, {}).get(f'{engine}-{method}')
            grid[(ds, engine, method)] = ({a: snap(cu, a) for a in alpha_tab}
                                          if cu else {a: (np.nan, np.nan) for a in alpha_tab})
    print('\n' + emit_latex_table(grid, alpha_tab))


if __name__ == '__main__':
    main()