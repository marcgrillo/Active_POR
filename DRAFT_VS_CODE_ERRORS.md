# $\mathcal{E}_{\text{link}}$ / $\mathcal{E}_{\text{est}}$ — Draft vs. Code: decisions & TODO

Reconciliation of the **Tripartite Error Framework** (Sec. `errorframework`, `paper_4.tex:1206`)
with the implementation. Captures the decisions made and what is left to do.

Key locations:
- Per-step driver / accumulation: [experiments/simulation.py](experiments/simulation.py) (~231–489)
- FTRL estimation-error diagnostic: [inference/engine.py:792](inference/engine.py#L792) `calculate_estimation_error_diagnostic`
- BAYES estimation-error diagnostic: [inference/engine.py:905](inference/engine.py#L905) `calculate_bayes_estimation_error`
- Specs the code follows (not yet in the paper): [todo.tex](todo.tex) (FTRL $\epsilon_t$), [update_delta.tex](update_delta.tex) (BAYES $\mathcal E_{\text{est}}$)

---

## Decision log

| # | Item | Decision | Where it lives |
|---|------|----------|----------------|
| 1 | Hybrid **BALD+US** martingale switch | **Own separate paper.** Keep in repo, remove from this paper. | code only |
| 2 | FTRL **$B_t$** inconsistency in $\mathcal E_{\text{link}}$ | **Fix the code** (done — see below). Paper formula was already correct. | code |
| 3 | $\epsilon_t$ is **coordinate-wise**, not $\sup$ | **Intentional.** Update the draft to match. | paper |
| 4 | BAYES $\mathcal E_{\text{est}}$ = MC equivalence-set diagnostic | **Add to paper.** Implementation audited & matches `update_delta.tex`. | paper |
| 4a | Confidence multiplier $c_t$ | **Fixed constant $c_t = 2$** (done). Not the $z_{1-\alpha/(M-1)}$ / time-uniform forms. | code |
| 4b | Number of batches $B$ | **$B=\min(20,\max(10,S/100))$, $\ge 2$ samples/batch** (done). | code |
| 4c | BALD+US corrupts the BAYES $\mathcal E_{\text{est}}$ | **Print a warning** naming the bad columns (done). | code |

---

## Code changes applied this session ✅

1. **FTRL $B_t$ now consistent with the realized surprise** — [simulation.py](experiments/simulation.py) FTRL branch.
   Was: $B_t$ = Laplace–Taylor closed-form BALD (`0.5·var_eta/(p(1-p))`) or `max(scores)`.
   Now: `B_t = p_t_1*S_t_1 + p_t_0*S_t_0`, the predictive average of the *same* Gaussian-KL
   surprises used for $S_t(y_t)$, with $p_t$ the MAP plug-in predictive probabilities.
   This matches the paper's $\widetilde B_t(q_t)=\sum_y \widehat p_t^{\mathrm{MAP}}(y)\,\widetilde S_t(y)$
   (near `paper_4.tex:1589`) and makes $\Delta_t = S_t(y_t)-B_t$ a genuine zero-mean martingale
   difference under correct specification. (The `var_eta` block and the `max(scores)` branch were
   removed.) The BALD acquisition for *selection* is unchanged.

2. **BAYES $\mathcal E_{\text{est}}$ multiplier & batches** — [engine.py:905](inference/engine.py#L905).
   `c_t = 2.0` (fixed); `B = min(20, max(10, S//100))` capped at `S//2`.

3. **BALD+US warning** — [engine.py:905](inference/engine.py#L905). One-shot `RuntimeWarning`:
   under `active_method='BALD+US'`, `_calculate_scores` flips between BALD and US per batch, so the
   saved columns `E_size` ($\Delta_t^{MC}$), `E_sep` (mcse) and the derived `|C_t|`/`rho_t` are
   **unreliable** for those runs.

> Both files re-checked with `py_compile` — OK.

---

## BAYES $\mathcal E_{\text{est}}$ audit (vs `update_delta.tex`) ✅ faithful

`q^\star=\arg\max\widehat A_t`; gap $\widehat G_t(q)=\widehat A_t(q^\star)-\widehat A_t(q)$; batch gaps
with $q^\star$ **fixed from the full sample**; batch-means MCSE $\mathrm{sd}(\cdot)/\sqrt B$;
equivalence set $\mathcal C_t=\{q:\widehat G_t\le c_t\,\mathrm{mcse}\}$; outputs $\rho_t$, $|\mathcal C_t|$,
$\Delta_t^{MC}$, median/max mcse. The old fixed-threshold ratio $R_{MC,t}$ is correctly **not** used.
$q^\star$ is always in $\mathcal C_t$ (its gap and mcse are 0). The only deviations from the spec are the
now-decided ones (4a/4b) plus dropping the optional time-uniform $\alpha_t$ schedule.

---

## Paper-writing TODO

- [ ] **(3) Coordinate-wise $\epsilon_t$.** Rewrite Eq. `relative_cubic_distortion` (`paper_4.tex:1753`):
      replace $\tfrac13\sup_{\|u\|=1}|T[Lu,Lu,Lu]|$ with the coordinate-wise
      $\widehat\epsilon_t^{\mathrm{coord}}=\tfrac13\max_j|D_{t,j}|$ over one-sigma directions $v_j=Le_j$.
      Add the BT-analytic vs LIN-finite-difference split and the regularizer cubic term (from `todo.tex`).
- [ ] **(4) New BAYES $\mathcal E_{\text{est}}$ subsection.** Document the MC equivalence-set diagnostic
      (`update_delta.tex`): $\widehat G_t$, batch-means MCSE, $\mathcal C_t^{MC}$, $\rho_t$, $\Delta_t^{MC}$.
      State the **fixed $c_t=2$** choice and the batch rule $B=\min(20,\max(10,S/100))$. Replace the paper's
      vague "reported in Appendix" stub (`paper_4.tex:1652`) with this.
- [ ] **(1) Remove the hybrid from this paper.** Drop the dangling `\ref{sec:bald_us_hybrid}` in the intro
      (`paper_4.tex:124`) and the commented contribution bullet (`paper_4.tex:156`); the BALD+US strategy
      moves to its own paper.
- [ ] **Appendix.** Either create `appendix.tex` (home for the sampling-error diagnostics
      `app:sampling_error_diagnostics` and, optionally, the BAYES $\mathcal E_{\text{est}}$ details) or
      remove `\input{appendix}` (`paper_4.tex:2153`).
- [ ] **Minor.** (a) Eq. `laplace_taylor_estimation_error` is a plain sum but the code sums absolute
      values $|\Delta_{\mathrm{geom}}|+|\Delta_{\mathrm{Tay}}|$ — align the two. (b) The relative
      $R_{\mathrm{est},t}$ (Eq. `relative_laplace_taylor_error`) is described but not logged — either log it
      or mark it as analysis-only. (c) Note that FTRL link KL is evaluated in ALR/tangent coordinates
      (dim $d-1$), consistent with the draft's tangent-space remark but written there with dim $d$.

## Code TODO (optional / low priority)

- [ ] Decide whether the unused `alpha` kwarg of `calculate_bayes_estimation_error` should be removed now
      that $c_t$ is constant.
- [ ] If `BALD+US` $\mathcal E_{\text{est}}$ is ever needed, make `_calculate_scores` take a fixed score
      type for the diagnostic instead of the per-call `corrcoef` switch.
