"""
Literature active-elicitation baselines for POR.

These are *query-selection rules* only: each returns a score per candidate pair so that
`argmax` picks the next query. The POR inference (FTRL/BAYES) and all metrics are unchanged,
so differences are attributable purely to the selection policy.

Strategies
----------
RANKUNC    Ranking-uncertainty heuristic. Picks the pair whose induced ordering probability
           (pairwise outranking index POI = P(U_i > U_j)) is closest to 0.5. Uses the POR
           posterior (BAYES) or Laplace samples (FTRL).
POLY       Polyhedral bisection (Toubia et al.). Picks the query whose hyperplane most evenly
           bisects the current feasible weight polytope.
CHEB       Chebyshev-center selection. Selects the query the analytic/Chebyshev center is most
           undecided about.
MAXREGRET  Minimax-regret query selection (Boutilier et al.).

The geometric/regret baselines operate on the ROR feasible polytope over the normalised weight
simplex, built from observed answers with UTA-style slack so it stays non-empty under
inconsistent responses (see FeasiblePolytope).
"""

import numpy as np
import cvxpy as cp

BASELINE_METHODS = {'RANKUNC', 'POLY', 'CHEB', 'MAXREGRET'}

# Methods that require a posterior distribution — N/A for FTRL (MAP only, no posterior).
_POSTERIOR_METHODS = {'RANKUNC'}


# ---------------------------------------------------------------------------
# Feasible weight polytope on the simplex, with UTA-style slack
# ---------------------------------------------------------------------------

class FeasiblePolytope:
    """
    Plausible-weight set used by the geometric/regret baselines:

        P = { w in Delta_d : r_k . w >= -eps_k*  for each observed answer k },

    where r_k = x_winner - x_loser and eps* is the minimal UTA-style slack that keeps P
    non-empty under inconsistent answers (eps* = 0 when the answers are consistent).
    All geometry is done on the normalised simplex, which is ranking-invariant, so LIN and
    BT share the same representation.
    """

    def __init__(self, R, d, delta=1e-3, solver=None):
        self.R = np.atleast_2d(R) if (R is not None and len(R)) else np.zeros((0, d))
        self.d = d
        self.k = self.R.shape[0]
        self.delta = delta                      # strict-preference margin: r_k . w >= delta
        self.solver = solver or cp.SCS
        self.eps = np.zeros(self.k)
        self.rhs = np.full(self.k, delta)       # effective lower bounds: R w >= rhs (= delta - eps*)
        self._w0 = None
        self._solve_slack()

    def _solve_slack(self):
        """Minimise total slack so that r_k . w >= delta - eps_k; cache eps* and an interior point."""
        if self.k == 0:
            self._w0 = np.full(self.d, 1.0 / self.d)
            return
        w = cp.Variable(self.d)
        eps = cp.Variable(self.k, nonneg=True)
        cons = [w >= 0, cp.sum(w) == 1, self.R @ w >= self.delta - eps]
        try:
            cp.Problem(cp.Minimize(cp.sum(eps)), cons).solve(solver=self.solver)
            if w.value is not None:
                self.eps = np.maximum(np.asarray(eps.value).ravel(), 0.0)
                self.rhs = self.delta - self.eps
                self._w0 = np.clip(np.asarray(w.value).ravel(), 0, None)
                s = self._w0.sum()
                self._w0 = self._w0 / s if s > 0 else np.full(self.d, 1.0 / self.d)
        except Exception:
            pass
        if self._w0 is None:
            self._w0 = np.full(self.d, 1.0 / self.d)

    def interior_point(self):
        return self._w0.copy()

    def chebyshev_center(self):
        """Chebyshev center of P (largest inscribed ball, full-space norms). Falls back to w0."""
        if self.k == 0:
            return self._w0.copy()
        w = cp.Variable(self.d)
        rad = cp.Variable(nonneg=True)
        norms = np.linalg.norm(self.R, axis=1)
        cons = [cp.sum(w) == 1,
                w >= rad,                                   # w_j - rad >= 0
                self.R @ w - self.rhs >= cp.multiply(norms, rad)]
        try:
            cp.Problem(cp.Maximize(rad), cons).solve(solver=self.solver)
            if w.value is not None:
                wc = np.clip(np.asarray(w.value).ravel(), 0, None)
                s = wc.sum()
                return wc / s if s > 0 else self._w0.copy()
        except Exception:
            pass
        return self._w0.copy()

    def _chord(self, w, dvec):
        """Interval [t_lo, t_hi] of t with w + t*dvec inside P."""
        t_lo, t_hi = -np.inf, np.inf
        # simplex non-negativity: w_j + t d_j >= 0
        for wj, dj in zip(w, dvec):
            if dj > 1e-12:
                t_lo = max(t_lo, -wj / dj)
            elif dj < -1e-12:
                t_hi = min(t_hi, -wj / dj)
        # answer half-spaces: (R w - rhs) + t (R d) >= 0
        if self.k:
            a = self.R @ w - self.rhs
            b = self.R @ dvec
            for ak, bk in zip(a, b):
                if bk > 1e-12:
                    t_lo = max(t_lo, -ak / bk)
                elif bk < -1e-12:
                    t_hi = min(t_hi, -ak / bk)
        return t_lo, t_hi

    def sample(self, n=400, burn=100, thin=2, seed=None):
        """Hit-and-run samples from P over the sum-zero subspace of the simplex."""
        rng = np.random.default_rng(seed)
        w = self.interior_point()
        out = []
        steps = burn + n * thin
        for t in range(steps):
            dvec = rng.standard_normal(self.d)
            dvec -= dvec.mean()                 # keep sum(w)=1
            nrm = np.linalg.norm(dvec)
            if nrm < 1e-12:
                continue
            dvec /= nrm
            t_lo, t_hi = self._chord(w, dvec)
            if not np.isfinite(t_lo) or not np.isfinite(t_hi) or (t_hi - t_lo) < 1e-12:
                continue
            w = w + rng.uniform(t_lo, t_hi) * dvec
            w = np.clip(w, 0, None)
            s = w.sum()
            if s > 0:
                w = w / s
            if t >= burn and (t - burn) % thin == 0:
                out.append(w.copy())
        if not out:
            out = [self.interior_point()]
        return np.array(out)

    def max_linear(self, c):
        """max_{w in P} c . w  (LP). Used by minimax-regret."""
        if self.k == 0:
            # over the simplex the max of a linear form is at a vertex
            return float(np.max(c))
        w = cp.Variable(self.d)
        cons = [w >= 0, cp.sum(w) == 1, self.R @ w >= self.rhs]
        try:
            cp.Problem(cp.Maximize(c @ w), cons).solve(solver=self.solver)
            if w.value is not None:
                return float(c @ np.asarray(w.value).ravel())
        except Exception:
            pass
        return float(c @ self._w0)


def _build_polytope(sampler):
    """Feasible polytope from the sampler's observed-answer diff vectors (winner - loser)."""
    R = sampler.X_diff if getattr(sampler, 'X_diff', None) is not None else None
    d = sampler.X.shape[1]
    return FeasiblePolytope(R, d)


# ---------------------------------------------------------------------------
# Sample source (shared with the information-theoretic methods)
# ---------------------------------------------------------------------------

def _get_samples(sampler, alg, current_state, n_samples):
    """
    Return a (S, d) array of plausible weight vectors.
      - BAYES: current_state already holds posterior samples.
      - FTRL : current_state is the MAP point; draw Laplace samples around it.
    """
    algo_type = alg.split('-')[0]
    if algo_type == 'BAYES':
        return np.atleast_2d(current_state)
    # FTRL
    omega_map = current_state
    Sigma = sampler.compute_laplace_covariance(omega_map, alg=alg)
    return sampler.sample_laplace(omega_map, Sigma, alg=alg, n_samples=n_samples)


def _candidate_diffs(sampler, candidates):
    """(n_cand, d) matrix of feature-difference vectors r_q = x_i - x_j."""
    idx_a = [c[0] for c in candidates]
    idx_b = [c[1] for c in candidates]
    return sampler.X[idx_a] - sampler.X[idx_b]


# ---------------------------------------------------------------------------
# RANKUNC — ranking-uncertainty heuristic
# ---------------------------------------------------------------------------

def rank_uncertainty(sampler, candidates, samples):
    """
    Score = -|POI_q - 0.5|, where POI_q = P(U_i > U_j) estimated over `samples`.
    Maximising selects the pair whose induced ordering is most uncertain.
    """
    vec_diff = _candidate_diffs(sampler, candidates)      # (n_cand, d)
    u_diff = vec_diff @ samples.T                         # (n_cand, S)
    poi = np.mean(u_diff > 0.0, axis=1)                   # (n_cand,)
    return -np.abs(poi - 0.5)


# ---------------------------------------------------------------------------
# POLY — polyhedral bisection (Toubia et al.)
# ---------------------------------------------------------------------------

def polyhedral_bisection(sampler, candidates, polytope, n_samples=400):
    """
    Score = -|f_q - 0.5|, where f_q is the fraction of feasible-polytope samples with
    r_q . w > 0. Maximising selects the query whose hyperplane most evenly bisects P.
    """
    W = polytope.sample(n=n_samples)                      # (S, d)
    vec_diff = _candidate_diffs(sampler, candidates)      # (n_cand, d)
    frac_pos = np.mean((vec_diff @ W.T) > 0.0, axis=1)    # (n_cand,)
    return -np.abs(frac_pos - 0.5)


# ---------------------------------------------------------------------------
# CHEB — Chebyshev-center selection
# ---------------------------------------------------------------------------

def chebyshev_selection(sampler, candidates, polytope):
    """
    Score = -|r_q . w_c|, where w_c is the Chebyshev center of P. Maximising selects the
    query the center is most undecided about (its hyperplane passes nearest the center).
    """
    w_c = polytope.chebyshev_center()                     # (d,)
    vec_diff = _candidate_diffs(sampler, candidates)      # (n_cand, d)
    return -np.abs(vec_diff @ w_c)


# ---------------------------------------------------------------------------
# MAXREGRET — max-regret pair selection (Boutilier et al.)
# ---------------------------------------------------------------------------

def max_regret(sampler, candidates, polytope, n_samples=400):
    """
    Score_q = max_{w in P} |U_w(i) - U_w(j)|, the largest remaining utility difference of the
    pair over the feasible polytope (its pairwise max regret). Maximising queries the pair with
    the largest unresolved utility swing. Estimated from hit-and-run samples of P, which avoids
    an O(m^2) LP solve per step.
    """
    W = polytope.sample(n=n_samples)                      # (S, d)
    U = sampler.X @ W.T                                   # (m, S) utilities per sample
    idx_a = [c[0] for c in candidates]
    idx_b = [c[1] for c in candidates]
    diff = U[idx_a] - U[idx_b]                            # (n_cand, S)
    return np.max(np.abs(diff), axis=1)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def is_applicable(active_method, alg):
    """Return False when the method is N/A for the given algorithm type."""
    algo_type = alg.split('-')[0]
    return not (algo_type == 'FTRL' and active_method in _POSTERIOR_METHODS)


def score(sampler, candidates, active_method, alg, current_state, n_samples):
    """Return a score array (one per candidate) for the requested baseline.
    Raises ValueError if the method is N/A for the given algorithm."""
    if not is_applicable(active_method, alg):
        raise ValueError(
            f"{active_method} is not applicable for {alg}: it requires a posterior "
            f"distribution which {alg.split('-')[0]} does not provide."
        )
    if active_method == 'RANKUNC':
        samples = _get_samples(sampler, alg, current_state, n_samples)
        return rank_uncertainty(sampler, candidates, samples)

    if active_method == 'POLY':
        return polyhedral_bisection(sampler, candidates, _build_polytope(sampler))

    if active_method == 'CHEB':
        return chebyshev_selection(sampler, candidates, _build_polytope(sampler))

    if active_method == 'MAXREGRET':
        return max_regret(sampler, candidates, _build_polytope(sampler))

    raise ValueError(f"Unknown baseline method: {active_method}")
