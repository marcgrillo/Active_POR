"""
Directly measure E_est: discrepancy between FTRL Laplace+Taylor acquisition scores
and an accurate importance-sampling (IS) reference, and correlate it with the
epsilon^Lap diagnostic. No link/signal confound -- both are estimators of the SAME
acquisition on the SAME posterior.

IS reference: draw from the PRIOR (Dirichlet for LIN, Gamma for BT), importance-weight
by the posterior. Valid in the few-data regime (posterior ~ prior); ESS is reported
and low-ESS states are filtered.

For each model (FTRL-LIN: epsilon often >1; FTRL-BT: control, epsilon<1), at several
early elicitation states n, for several HMs:
  - epsilon^Lap
  - BALD^LT (bald_mi_linear_appr) vs BALD^MC (IS)   -> ordering discrepancy
  - US^LT  (plug-in MAP entropy) vs US^MC  (IS)      -> ordering discrepancy
Discrepancy = 1 - Kendall tau over candidates (+ top-1 disagreement). Expect a
positive correlation between epsilon^Lap and the BALD discrepancy; US (covariance-free)
should stay low for LIN regardless of epsilon.
"""
import os, argparse
import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import kendalltau, spearmanr
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import utils
from mcda.models import PiecewiseLinearTransformer
from inference.engine import PreferenceSampler

DS = 'exp_dataset'
F3 = 100


def binary_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def is_mc_scores(sampler, model, candidates, n_samples=40000, seed=0):
    """IS-from-prior estimate of US and BALD scores for all candidates. Returns
    (US_mc, BALD_mc, ess_frac)."""
    rng = np.random.default_rng(seed)
    X = sampler.X
    d = sampler.n_params
    prefs = sampler.prefs
    if prefs.shape[0] == 0:
        return None, None, 0.0
    Xd = X[prefs[:, 0]] - X[prefs[:, 1]]   # winner - loser

    if model == 'LIN':
        samples = rng.dirichlet(np.ones(d), size=n_samples)          # flat proposal
        u = Xd @ samples.T
        p = np.clip(0.5 * (1.0 + u), 1e-12, 1 - 1e-12)
        loglik = np.sum(np.log(p), axis=0)
        logprior = sampler.lambda_dirichlet_ftrl_lin * np.sum(np.log(samples + 1e-12), axis=1)
        logw = loglik + logprior                                     # proposal ~ uniform
    else:  # BT, prior = proposal = Gamma(alpha,beta)
        a = sampler.gamma_alpha_ftrl_bt; b = sampler.gamma_beta_ftrl_bt
        samples = rng.gamma(shape=a, scale=1.0 / b, size=(n_samples, d))
        u = Xd @ samples.T
        loglik = np.sum(-np.logaddexp(0.0, -u), axis=0)
        logw = loglik                                                # prior cancels proposal

    logw -= logsumexp(logw)
    w = np.exp(logw)
    ess = 1.0 / np.sum(w ** 2) / n_samples

    idx_a = [c[0] for c in candidates]; idx_b = [c[1] for c in candidates]
    vd = X[idx_a] - X[idx_b]
    ud = vd @ samples.T
    probs = expit(ud) if model == 'BT' else np.clip(0.5 * (1 + ud), 1e-12, 1 - 1e-12)
    p_mean = probs @ w
    US_mc = binary_entropy(p_mean)
    H_cond = binary_entropy(probs) @ w
    BALD_mc = US_mc - H_cond
    return US_mc, BALD_mc, float(ess)


def lt_scores(sampler, alg, w_map, Sigma, candidates):
    """Laplace+Taylor analytic US (plug-in MAP entropy) and BALD scores."""
    model = alg.split('-')[1]
    idx_a = [c[0] for c in candidates]; idx_b = [c[1] for c in candidates]
    vd = sampler.X[idx_a] - sampler.X[idx_b]
    t = vd @ w_map
    eta = expit(t) if model == 'BT' else np.clip(0.5 * (1 + t), 1e-12, 1 - 1e-12)
    US_lt = binary_entropy(eta)                                  # plug-in MAP US
    BALD_lt = sampler.bald_mi_linear_appr(w_map, Sigma, candidates, model)
    return US_lt, np.asarray(BALD_lt)


def run(alg, f1, f2, hms, steps, n_samples, ess_min=0.02):
    tables, rankings, dm_prefs, Us = utils.read_dataset(os.path.join('datasets', DS), f1, f2, F3)
    model = alg.split('-')[1]
    rows = []
    for hm in hms:
        transformer = PiecewiseLinearTransformer.from_equal_intervals(tables[hm], num_intervals=3)
        Xf = transformer.transform(tables[hm])
        prefs = dm_prefs[hm]
        sampler = PreferenceSampler(Xf, [], transformer.total_params)
        for n in range(1, max(steps) + 1):
            a, b = prefs[n - 1]
            sampler.add_preference(a, b)
            if n not in steps:
                continue
            w_map = sampler.find_map(model=model)
            Sigma = sampler.compute_laplace_covariance(w_map, alg=alg)
            diag = sampler.calculate_estimation_error_diagnostic(w_map, Sigma, [(0, 1)], alg, 'BALD')
            eps = diag['epsilon_t']
            seen = set(tuple(x) for x in sampler.prefs)
            cands = [p for p in itertools.combinations(range(f1), 2)
                     if p not in seen and (p[1], p[0]) not in seen]
            if len(cands) < 5:
                continue
            US_mc, BALD_mc, ess = is_mc_scores(sampler, model, cands, n_samples, seed=hm * 100 + n)
            if US_mc is None or ess < ess_min:
                continue
            US_lt, BALD_lt = lt_scores(sampler, alg, w_map, Sigma, cands)
            # ordering discrepancy = 1 - Kendall tau
            tb = kendalltau(BALD_lt, BALD_mc).correlation
            tu = kendalltau(US_lt, US_mc).correlation
            # top-1 disagreement: rank of LT-argmax in MC order (0=best)
            rb = int(np.sum(BALD_mc > BALD_mc[np.argmax(BALD_lt)])) / len(cands)
            ru = int(np.sum(US_mc > US_mc[np.argmax(US_lt)])) / len(cands)
            # information-gain regret of the BALD (Laplace) selection vs the true best:
            # fraction of the maximum available information gain forfeited.
            BALD_mc_pos = np.clip(BALD_mc, 0, None)
            qlt = int(np.argmax(BALD_lt))
            mx = float(np.max(BALD_mc_pos))
            bald_regret = float(1.0 - BALD_mc_pos[qlt] / mx) if mx > 1e-6 else np.nan
            rows.append(dict(alg=alg, f1=f1, f2=f2, hm=hm, n=n, epsilon=eps, ess=ess,
                             bald_disc=1 - (tb if tb == tb else 0),
                             us_disc=1 - (tu if tu == tu else 0),
                             bald_top1_miss=rb, us_top1_miss=ru,
                             bald_regret=bald_regret))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--f1', type=int, default=10)
    ap.add_argument('--f2', type=int, nargs='+', default=[2, 6])
    ap.add_argument('--hms', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--steps', type=int, nargs='+', default=[1, 3, 5, 8, 11, 14, 18, 22, 27, 33])
    ap.add_argument('--nsamp', type=int, default=40000)
    args = ap.parse_args()

    import pandas as pd
    allrows = []
    for alg in ['FTRL-LIN', 'FTRL-BT']:
        for f2 in args.f2:
            allrows += run(alg, args.f1, f2, args.hms, args.steps, args.nsamp)
    df = pd.DataFrame(allrows)
    if df.empty:
        print('No usable points (ESS too low?).'); raise SystemExit
    df.to_csv(os.path.join('scratch', 'est_discrepancy.csv'), index=False)
    pd.set_option('display.float_format', lambda v: f'{v:.3f}')
    print(f'\nUsable points: {len(df)} (ESS>=2%)')
    print('\n=== Mean by alg and epsilon regime ===')
    df['eps_gt1'] = df['epsilon'] > 1
    print(df.groupby(['alg', 'eps_gt1']).agg(
        n=('hm', 'count'), epsilon=('epsilon', 'mean'), ess=('ess', 'mean'),
        bald_disc=('bald_disc', 'mean'), us_disc=('us_disc', 'mean'),
        bald_top1_miss=('bald_top1_miss', 'mean')).to_string())

    print('\n=== Correlation epsilon^Lap vs discrepancy ===')
    for col in ['bald_disc', 'us_disc', 'bald_top1_miss']:
        rho, p = spearmanr(df.epsilon, df[col])
        print(f'  {col}: spearman={rho:+.3f} (p={p:.1e})')

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for alg, c in [('FTRL-LIN', 'tab:red'), ('FTRL-BT', 'tab:blue')]:
        s = df[df.alg == alg]
        ax.scatter(s.epsilon, s.bald_disc, c=c, marker='o', alpha=0.6, s=40, label=f'{alg} BALD')
        ax.scatter(s.epsilon, s.us_disc, c=c, marker='x', alpha=0.4, s=40, label=f'{alg} US')
    ax.axvline(1, ls='--', c='gray')
    ax.set_xlabel(r'$\widehat\epsilon_t^{\mathrm{Lap}}$')
    ax.set_ylabel(r'score discrepancy  $1-\tau$(Laplace-Taylor, IS-MC)')
    ax.set_title('Laplace+Taylor acquisition error grows with epsilon^Lap\n(affects BOTH BALD and US scores; FTRL-BT control stays low)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join('scratch', 'est_discrepancy.png')
    fig.savefig(out, dpi=140); print('\nwrote', out)