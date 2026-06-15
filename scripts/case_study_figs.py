"""
Build the case-study figure (marginal value functions at t=1,3,5) and the final-RAI table
from the saved per-step value-function states in scratch/case_study/<variant>/{t}_active.npy.

Marginals: with num_intervals=3 the parameter vector w has 3 segment weights per criterion.
For criterion j and value x, marginal_j(x) = sum_k w[3j+k] * clip((x-b_k)/(b_{k+1}-b_k),0,1),
with breakpoints b = linspace(min_j, max_j, 4). Each value function is normalized so the
best-possible alternative has total utility 1 (sum of weights = 1), making shapes comparable.
RAI: rank every posterior sample (BAYES) -> graded; FTRL uses its single MAP ranking -> 100/0.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from case_study_run import TABLE, NAMES, VARIANTS

OUT = os.path.join('scratch', 'case_study')
NINT = 3
CRIT = ['$g_1$ (system)', '$g_2$ (access)', '$g_3$ (flagship)', '$g_4$ (economy)']
BP = [np.linspace(TABLE[:, j].min(), TABLE[:, j].max(), NINT + 1) for j in range(4)]
STEPS = [1, 10, 30]
CASE = ['BAYES-BT', 'FTRL-BT']      # case study illustrates the BT link only


def marginals(w, j, xgrid):
    """marginal value of criterion j over xgrid for one weight vector w (len 12)."""
    b = BP[j]
    feat = np.stack([np.clip((xgrid - b[k]) / (b[k + 1] - b[k]), 0, 1) for k in range(NINT)], axis=1)
    return feat @ w[j * NINT:(j + 1) * NINT]


def load_w(alg, t):
    return np.load(os.path.join(OUT, alg, f'{t}_active.npy'))


def is_bayes(alg):
    return alg.startswith('BAYES')


# ---------------------------------------------------------------- figure
BUDGET = 12
TMAX = 15


def load_track(d, t, active):
    p = os.path.join(OUT, d, f'{t}_active.npy' if active else f'{t}.npy')
    return np.load(p)


def rai_track(d, t, active):
    w = load_track(d, t, active)
    from mcda.models import PiecewiseLinearTransformer
    tr = PiecewiseLinearTransformer.from_equal_intervals(TABLE, NINT)
    X = tr.transform(TABLE)
    W = np.atleast_2d(w); U = X @ W.T
    rc = np.zeros((len(TABLE), len(TABLE)))
    for s in range(U.shape[1]):
        rc[np.arange(len(TABLE)), np.argsort(np.argsort(-U[:, s]))] += 1
    return rc / U.shape[1]


def make_figure():
    """BALD vs US vs random elicitation: how fast the rank-acceptability distributions
    concentrate, for the sampling-based learner BAYES-BT."""
    ts = list(range(1, TMAX + 1))
    # active = BALD and US; random = passive track (identical across runs, take BALD's)
    series = {'BALD': ('BAYES-BT_BALD', True), 'US': ('BAYES-BT_US', True),
              'random': ('BAYES-BT_BALD', False)}
    modal = {k: [] for k in series}
    toppick = {k: [] for k in series}
    for t in ts:
        for k, (d, active) in series.items():
            R = rai_track(d, t, active)
            modal[k].append(float(np.mean(np.max(R, axis=1))))
            toppick[k].append(float(np.max(R[:, 0])))
    style = {'BALD': ('-o', '#2471a3'), 'US': ('-^', '#e67e22'), 'random': ('-s', '#c0392b')}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, data, ttl, yl in (
        (ax[0], modal, 'Overall ranking concentration', r'mean modal RAI  $\frac{1}{n}\sum_i\max_r\mathrm{RAI}(a_i,r)$'),
        (ax[1], toppick, 'Confidence in the top alternative', r'$\max_i \mathrm{RAI}(a_i,\,1)$')):
        for k in ('BALD', 'US', 'random'):
            ls, c = style[k]
            a.plot(ts, data[k], ls, ms=4, color=c, label=k)
        a.axvline(BUDGET, ls='--', color='gray', lw=1)
        a.text(BUDGET - 0.2, a.get_ylim()[1], f'budget $T={BUDGET}$', color='gray',
               fontsize=8, va='top', ha='right')
        a.set_xlabel('elicitation step $t$'); a.set_ylabel(yl, fontsize=10)
        a.set_title(ttl, fontsize=11); a.grid(alpha=0.25)
    ax[1].legend(fontsize=9, loc='lower left')
    fig.tight_layout()
    p = os.path.join('figs', 'case_study_active_vs_random.png')
    os.makedirs('figs', exist_ok=True)
    fig.savefig(p, dpi=150); print('wrote', p)


# ---------------------------------------------------------------- RAI
def rai_matrix(alg, t):
    """RAI[i, r] = P(alt i at rank r). BAYES: over samples; FTRL: single MAP ranking."""
    from mcda.models import PiecewiseLinearTransformer
    tr = PiecewiseLinearTransformer.from_equal_intervals(TABLE, NINT)
    X = tr.transform(TABLE)
    w = load_w(alg, t)
    n = len(TABLE)
    W = np.atleast_2d(w)                 # (S,12) or (1,12)
    U = X @ W.T                          # (n, S)
    rc = np.zeros((n, n))
    for s in range(U.shape[1]):
        ranks = np.argsort(np.argsort(-U[:, s]))
        rc[np.arange(n), ranks] += 1
    return rc / U.shape[1]


def make_rai_table(t):
    """BT learner. Rows ordered by FTRL-BT's own (deterministic) recommended ranking; FTRL-BT
    assigns RAI=1 to each attained rank, BAYES-BT grades it, shown under BALD and US."""
    from mcda.models import PiecewiseLinearTransformer
    tr = PiecewiseLinearTransformer.from_equal_intervals(TABLE, NINT)
    X = tr.transform(TABLE)
    rb = rai_track('BAYES-BT_BALD', t, True)
    ru = rai_track('BAYES-BT_US', t, True)
    wf = load_track('FTRL-BT_BALD', t, True)            # FTRL MAP -> deterministic ranking
    ftrl_rank = np.argsort(np.argsort(-(X @ wf)))        # rank (0-based) of each alternative
    order = np.argsort(ftrl_rank)[:6]
    print(f'\n=== Final RAI at t={t} (rows = FTRL-BT recommended order) ===')
    print(f'{"Country":8} {"rank":4} {"FTRL":6} {"BAYES/BALD":10} {"BAYES/US":9}')
    rows = []
    for i in order:
        r = int(ftrl_rank[i])
        print(f'{NAMES[i]:8} {r+1:4d} {1.00:6.2f} {rb[i,r]:10.2f} {ru[i,r]:9.2f}')
        rows.append((NAMES[i], r, rb[i, r], ru[i, r]))
    return rows


if __name__ == '__main__':
    make_figure()
    make_rai_table(BUDGET)
