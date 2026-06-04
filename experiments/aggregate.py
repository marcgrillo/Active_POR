"""
Sample-efficiency aggregation for the experimental section.

From the per-step metric curves (ASRS/ASPS/AIOS, saved under tests/...), compute:
  - AULC      : area under the learning curve = mean_t metric_t
  - T_gamma   : queries needed to first reach a target metric value gamma
  - Saving    : 1 - T_gamma(method) / T_gamma(Random)
  - dAULC     : AULC(method) - AULC(Random)

Each strategy is a `sub_fold` (e.g. 'FTRL-BT_BALD'); its 'active' track is the strategy
curve and its 'passive' track is the Random baseline. Curves are (T, n_HM) matrices.

Usage:
    python -m experiments.aggregate --dataset test_dataset --f1 20 --f2 4 \
        --metric asrs --gamma 0.8 \
        --strategies FTRL-BT_BALD FTRL-BT_US
"""

import numpy as np
import pandas as pd

from common.utils import load_test_results


# ---------------------------------------------------------------------------
# Curve loading
# ---------------------------------------------------------------------------

def load_curve(metric, dataset_fold, sub_fold, f1, f2, f3, num_dm_dec, track='active'):
    """Return the (T, n_HM) curve for one strategy/metric. track in {'active','passive'}."""
    y_pass, y_act = load_test_results(metric, dataset_fold, sub_fold, num_dm_dec, f1, f2, f3)
    if y_pass.size == 0:
        return np.empty((0, 0))
    return y_act if track == 'active' else y_pass


# ---------------------------------------------------------------------------
# Per-HM sample-efficiency quantities
# ---------------------------------------------------------------------------

def aulc_per_hm(curve):
    """Per-HM AULC = mean over steps. curve (T, n_HM) -> (n_HM,)."""
    if curve.size == 0:
        return np.array([])
    return np.nanmean(curve, axis=0)


def t_gamma_per_hm(curve, gamma, cap_unreached=True):
    """
    Per-HM first step (1-indexed) at which the metric reaches `gamma`.
    If never reached: capped at T (budget) when cap_unreached else NaN.
    curve (T, n_HM) -> (n_HM,).
    """
    if curve.size == 0:
        return np.array([])
    T, n_hm = curve.shape
    out = np.full(n_hm, np.nan)
    for h in range(n_hm):
        hit = np.where(curve[:, h] >= gamma)[0]
        if len(hit):
            out[h] = hit[0] + 1
        elif cap_unreached:
            out[h] = T
    return out


def _mean_ci(vec):
    """Mean and 95% CI half-width (2*SEM) ignoring NaN."""
    vec = vec[~np.isnan(vec)]
    if vec.size == 0:
        return np.nan, np.nan
    mean = vec.mean()
    ci = 2 * vec.std(ddof=1) / np.sqrt(vec.size) if vec.size > 1 else 0.0
    return mean, ci


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def build_table(dataset_fold, strategies, f1, f2, f3, num_dm_dec,
                metric='asrs', gamma=0.8, random_sub_fold=None):
    """
    Build a tidy DataFrame with AULC, dAULC, T_gamma and Saving for each strategy.

    random_sub_fold: sub_fold whose ACTIVE track is the Random baseline. If None, the
    PASSIVE track of the first strategy is used as Random.
    """
    # --- Random reference ---
    if random_sub_fold is not None:
        rand_curve = load_curve(metric, dataset_fold, random_sub_fold, f1, f2, f3, num_dm_dec, 'active')
    else:
        rand_curve = load_curve(metric, dataset_fold, strategies[0], f1, f2, f3, num_dm_dec, 'passive')

    rand_aulc = aulc_per_hm(rand_curve)
    rand_tg = t_gamma_per_hm(rand_curve, gamma)
    rand_aulc_mean, _ = _mean_ci(rand_aulc)
    rand_tg_mean, _ = _mean_ci(rand_tg)

    rows = []
    # Random row first
    rows.append(dict(
        strategy='Random', metric=metric,
        AULC=rand_aulc_mean, AULC_CI=_mean_ci(rand_aulc)[1],
        dAULC=0.0,
        T_gamma=rand_tg_mean, Saving=0.0,
    ))

    for sf in strategies:
        curve = load_curve(metric, dataset_fold, sf, f1, f2, f3, num_dm_dec, 'active')
        if curve.size == 0:
            rows.append(dict(strategy=sf, metric=metric, AULC=np.nan, AULC_CI=np.nan,
                             dAULC=np.nan, T_gamma=np.nan, Saving=np.nan))
            continue
        a = aulc_per_hm(curve)
        tg = t_gamma_per_hm(curve, gamma)
        a_mean, a_ci = _mean_ci(a)
        tg_mean, _ = _mean_ci(tg)
        saving = (1 - tg_mean / rand_tg_mean) if (rand_tg_mean and not np.isnan(rand_tg_mean)) else np.nan
        rows.append(dict(
            strategy=sf, metric=metric,
            AULC=a_mean, AULC_CI=a_ci,
            dAULC=a_mean - rand_aulc_mean,
            T_gamma=tg_mean, Saving=saving,
        ))

    return pd.DataFrame(rows)


def to_latex(df, caption='AULC and sample efficiency.', label='tab:aulc'):
    """Render the table as a booktabs LaTeX table (best AULC bolded)."""
    d = df.copy()
    best = d['AULC'].idxmax()
    def fmt(r):
        a = f"{r.AULC:.3f}$\\pm${r.AULC_CI:.3f}" if not np.isnan(r.AULC) else '--'
        if r.name == best:
            a = f"\\textbf{{{a}}}"
        tg = '--' if np.isnan(r.T_gamma) else f"{r.T_gamma:.1f}"
        sv = '--' if np.isnan(r.Saving) else f"{100*r.Saving:.0f}\\%"
        da = '--' if np.isnan(r.dAULC) else f"{r.dAULC:+.3f}"
        name = str(r.strategy).replace('_', r'\_')
        return f"{name} & {a} & {da} & {tg} & {sv} \\\\"
    body = "\n".join(fmt(r) for _, r in d.iterrows())
    return (
        "\\begin{table}[t]\n\\centering\\footnotesize\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Strategy & AULC & $\\Delta$AULC & $T_\\gamma$ & Saving \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Aggregate sample-efficiency metrics.')
    p.add_argument('--dataset', required=True)
    p.add_argument('--f1', type=int, required=True)
    p.add_argument('--f2', type=int, required=True)
    p.add_argument('--f3', type=int, default=100)
    p.add_argument('--metric', default='asrs', choices=['asrs', 'asps', 'aios'])
    p.add_argument('--gamma', type=float, default=0.8)
    p.add_argument('--strategies', nargs='+', required=True,
                   help='sub_fold names, e.g. FTRL-BT_BALD FTRL-BT_US')
    p.add_argument('--random_sub_fold', default=None)
    p.add_argument('--num_dm_dec', type=int, default=None,
                   help='Number of steps. Defaults to round(f3*f1*(f1-1)/200).')
    p.add_argument('--latex', action='store_true', help='Also print a LaTeX table.')
    args = p.parse_args()

    ndm = args.num_dm_dec or int(round(args.f3 * args.f1 * (args.f1 - 1) / 200))

    df = build_table(args.dataset, args.strategies, args.f1, args.f2, args.f3, ndm,
                     metric=args.metric, gamma=args.gamma,
                     random_sub_fold=args.random_sub_fold)
    pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print(df.to_string(index=False))
    if args.latex:
        print('\n' + to_latex(df, caption=f'{args.metric.upper()} sample efficiency '
                                          f'(f1={args.f1}, f2={args.f2}, $\\gamma$={args.gamma}).'))
