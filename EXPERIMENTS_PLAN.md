# Experimental Plan — Active Learning for POR

Plan for the `\section{Experimental Evaluation}` of `paper_4.tex`. Grounded in what the
code can produce, with the new components flagged. Decisions locked in:
**(1)** implement all 4 literature baselines, **(2)** match the previous POR (EJOR 2025)
scale, **(3)** cross both misspecification axes to drive the $\mathcal E_{\text{link}}/\mathcal E_{\text{util}}$ story.

---

## 1. Factor design

Two swept factors ($F_1, F_2$), per `tab:test_info`. γ and inconsistency are held fixed /
treated as a binary condition rather than numbered factors.

| Paper factor | Meaning | Generator knob | Levels |
|---|---|---|---|
| $F_1$ | # alternatives $m$ | `F1` | **{10, 25, 50}** |
| $F_2$ | # criteria $n$ | `F2` | **{3, 7, 10}** |
| (fixed) | breakpoints γ | `num_intervals` | **γ = 3** (3 segments, 4 characteristic points) |
| condition | inconsistency | `target_inconsistency` | **{0.0 (consistent), 0.2 (inconsistent)}** _(0.2 confirm)_ |
| repetitions | Human Models | `hm` | **FTRL 20, BAYES 5** |

- Budget is **not** a factor (the old $F_3$ = data-percentage is retired). Query count is the
  x-axis. The elicitation budget $T$ is a **configurable cap**, not necessarily all pairs —
  active learning's value is in the early regime, so we can stop well before 100%. Proposed
  default: cap at a fraction (e.g. ~40% of pairs) or a multiple of $m$, chosen so curves
  plateau/separate. Big win at $m=50$ (cap ~490 steps vs 1225). _(cap value to confirm)_
- Full $F_1\times F_2$ grid = **9 dataset configs**, each with consistent + inconsistent versions.
- **Model dimension** $d = n\cdot\gamma = n\cdot 3$: ranges from $3\cdot3=9$ ($n=3$) up to
  $10\cdot3=30$ ($n=10$). The $n=10$ corner is a 30-dim simplex/orthant — heavy for BAYES MCMC.
- **Step count** $T = \text{round}(m(m-1)/2)$ at `F3=100`: $m=10\to45$, $m=25\to300$, $m=50\to1225$.
  The $m=50$ corner is 1225 sequential steps per run.

> **Draft edit needed:** `paper_4.tex` §Data Generation currently names families
> $F_1,F_2,F_3,F_{\text{inc}}$. Update to match `tab:test_info`: only $F_1$ (alternatives) and
> $F_2$ (criteria) are factors; consistency is a binary condition; γ is fixed.

### Misspecification grid (drives the error framework)

The generator's two knobs deliberately create well- vs mis-specified conditions:

| Knob | Value | Effect |
|---|---|---|
| `prob_model` | `logit` | matches **BT** link → $\mathcal E_{\text{link}}=0$ for BT, $>0$ for LIN |
| `prob_model` | `linear` | matches **LIN** link → $\mathcal E_{\text{link}}=0$ for LIN, $>0$ for BT |
| `utility_type` | `piecewise_linear` | matches POR utility → $\mathcal E_{\text{util}}\approx 0$ |
| `utility_type` | `exponential` | smooth truth → $\mathcal E_{\text{util}}>0$ |

- **H1 runs (main learning curves):** well-specified link + PWL utility (consistent &
  inconsistent). Produces the draft's 4-panel figure (LIN/BT × consistent/inconsistent).
- **H2 / error study:** sweep the full 2×2 misspecification grid so the BALD−US scatter
  spans a real range of $\mathcal E_{\text{link}}, \mathcal E_{\text{est}}$.

---

## 2. Elicitation strategies (7)

| Strategy | Status | Implementation approach |
|---|---|---|
| Random (PASSIVE) | ✅ | existing |
| Uncertainty Sampling | ✅ | existing — max predictive entropy |
| BALD | ✅ | existing — max MI |
| **Polyhedral bisection** | ❌ build | Maintain the ROR feasible polytope $\{w\in\mathcal W:\text{consistent with answers}\}$; pick the query whose hyperplane $w^\top r_q=0$ most evenly bisects it (approx via hit-and-run samples → pair with split closest to 50/50 under the uniform-on-polytope measure). |
| **Chebyshev-center** | ❌ build | LP for the Chebyshev center $w_c$ of the feasible polytope; select the query most ambiguous at $w_c$ (\|$w_c^\top r_q$\| smallest), i.e. split around the center. |
| **Ranking-uncertainty** | ❌ build (cheap) | Use POI/RAI already computed: pick the pair whose pairwise winning index is closest to 0.5 (most uncertain induced ordering). |
| **Max-regret** | ❌ build (costly) | Minimax-regret selection (Boutilier-style): for each candidate, estimate reduction in max regret of the recommended alternative over the feasible set; query the maximiser. One LP per candidate per step. |

**Fairness rule:** all 7 select queries on the *same* simulated answers, then are evaluated
with the *same* POR posterior metrics. The 4 deterministic baselines select using the
hard-constraint polytope from observed answers; POR (US/BALD) selects using its
posterior/Laplace state. This isolates the selection rule.

New module proposed: `inference/baselines.py` (polytope representation + 4 selectors),
wired into `get_candidate_scores` via new `active_method` values
`POLY`, `CHEB`, `RANKUNC`, `MAXREGRET`.

---

## 3. POR models (4)

FTRL-LIN, FTRL-BT, BAYES-LIN, BAYES-BT — all existing. Each runs every strategy.

---

## 4. Run matrix → outputs

| Figure / Table (draft) | Runs needed | New code |
|---|---|---|
| **ASRS learning curves** (4 panels: LIN/BT × cons/incons) | 7 strategies × {LIN,BT} × {cons, incons}, well-specified, averaged over $F_1$ & HMs | aggregation/plot |
| **AULC-ASRS table** | same runs, integrate curve | `AULC = mean_t ASRS_t` |
| **$T_\gamma$ / Saving table** | same runs | first-crossing of target $\gamma$ |
| **Sensitivity to $F_1,F_2,F_3,F_{\text{inc}}$** | sweep one factor at a time, others averaged | $\Delta$AULC vs Random |
| **BALD−US vs error scatter** | BALD & US over the **misspecification grid**; collect $\mathcal E_{\text{link}}, \mathcal E_{\text{est}}$ (already logged to `error_scores.csv`) | aggregation script (reuse `experiments/plot_diagnostics.py` loaders) |
| **Stopping criteria table** | re-use H1 runs; apply both rules post-hoc | (a) acquisition moving-avg < thresh for $k$ rounds; (b) confidence $C_t$ > thresh — both computable from saved per-step state |
| ASPS / AIOS (appendix) | same runs, different metric | existing `compute_asps/aios` |

---

## 5. New code checklist

- [ ] `inference/baselines.py`: feasible-polytope rep + `POLY`, `CHEB`, `RANKUNC`, `MAXREGRET` selectors.
- [ ] Wire the 4 selectors into `get_candidate_scores` / `simulation.py` dispatch.
- [ ] `experiments/aggregate.py`: AULC, $T_\gamma$, Saving$_\gamma$, $\Delta$AULC from the saved metric `.npy` curves.
- [ ] BALD−US-vs-error scatter script (extend `plot_diagnostics.py`).
- [ ] Two stopping-criterion evaluators (post-hoc over saved acquisition scores / $C_t$).
- [ ] Optional: a `compute_error_diagnostics` on/off flag in `simulation.py` so the bulk H1
      runs can skip the expensive FTRL hypothetical refits (only the H2 grid needs them).

---

## 6. Compute reality (flag)

Run count per condition: HM × strategies × POR-models × ($F_1\times F_2$ grid).
With FTRL HM=20, BAYES HM=5, 7 strategies, 9 configs:
- FTRL: 20 × 7 × 2 models × 9 = **2520 runs**; BAYES: 5 × 7 × 2 × 9 = **630 runs** — per consistency
  condition, before the H2 misspecification grid.
- Cost is dominated by the high corners: $m=50$ (1225 steps/run) and $n=10$ ($d=30$, slow MCMC).

Recommendations:
- Split **H1 bulk runs** (diagnostics OFF, fast) from the **H2 error grid** (diagnostics ON;
  BALD & US only; the misspec 2×2). **[agreed]**
- The 4 new baselines add LP/sampling per step — `MAXREGRET` is the bottleneck (LP per candidate),
  worst at $m=50$.
- Consider capping the $m=50 \times n=10$ corner (e.g. fewer HMs there) if runtime explodes.
- Use the existing `num_cores` multiprocessing; budget overnight+ per major sweep.

---

## 7. Settings

**Locked:** $F_1\in\{10,25,50\}$, $F_2\in\{3,7,10\}$, γ=3, HM = 20 (FTRL) / 5 (BAYES),
diagnostics-split H1/H2, all 4 baselines, cross both misspecification axes.

**Still to confirm:**
1. Inconsistent level (proposed 0.2).
2. Target $\gamma$ value(s) for the $T_\gamma$ / Saving table (note: $\gamma$ here is the ASRS
   target, not the breakpoint count).
3. Stopping-criterion thresholds and the consecutive-round count $k$.
4. Whether to cap HMs at the $m=50,\,n=10$ corner.
