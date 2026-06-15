# Paper scripts

Standalone scripts that regenerate the figures, tables, and supporting data for the
experimental and case-study sections. They are **not** part of the importable package
(`common/`, `experiments/`, `inference/`, `mcda/`); they drive it.

## Running

Run from the **repository root** so that the relative data paths (`datasets/`,
`samples/`, `tests/`, `scratch/`, `figs/`) resolve, with the repo root on `PYTHONPATH`
(the scripts import the package and, in a few cases, one another):

```bash
# Linux/macOS
PYTHONPATH=. python scripts/case_study_figs.py
```
```powershell
# Windows PowerShell
$env:PYTHONPATH="."; python scripts/case_study_figs.py
```

Generated outputs (`figs/*.png`, `*.npy`, `*.tex`, run data under `scratch/`, `samples/`,
`tests/`) are git-ignored; only the scripts themselves are tracked.

## What each script does

### Illustrative case study (20 European countries)
- **`case_study_run.py`** — runs the case study (BAYES-BT / FTRL-BT under BALD, US, random)
  to a fixed budget `T`; saves per-step value-function states under `scratch/case_study/`.
- **`case_study_figs.py`** — builds the active-vs-random concentration figure
  `figs/case_study_active_vs_random.png` and prints the final-RAI summary. *(imports `case_study_run`)*
- **`case_study_rai_table.py`** — emits the full rank-acceptability distribution table
  (`tab:final_rais`). *(imports `case_study_run`)*

### Performance / sensitivity / efficiency
- **`plot_sensitivity.py`** — AULC-gain sensitivity to `F1`, `F2`
  (`figs/sensitivity_asrs_asps_f1f2.png`).
- **`make_efficiency_table.py`** — sample-efficiency table (`tab:sample_efficiency`).
  *(imports `stopping_eval`)*

### Stopping rule
- **`stopping_eval.py`** — core stopping-rule evaluation: KL-divergence convergence signal,
  relative-threshold rule, net-benefit NB(α), with a raw-results cache. Imported by the
  scripts below.
- **`plot_nb.py`** — net-benefit `NB(α)` figure (`figs/stopping_nb_alpha.png`).
  *(imports `stopping_eval`, `plot_eta_gain`)*
- **`make_nb_table.py`** — stopping operating-characteristics table (`tab:stopping_curve`).
  *(imports `stopping_eval`, `plot_eta_gain`)*
- **`plot_eta_gain.py`** — efficiency-ratio helper used by the two scripts above.
  *(imports `stopping_eval`)*

### Link-misspecification diagnostic
- **`compute_table_values.py`** — computes `tab:link_dissociation` and `tab:link_cost`
  (mean ±2 SEM, Wilcoxon p, ceiling-normalized). *(imports `plot_h2`)*
- **`plot_h2.py`** — collects the `E_link` diagnostic across runs. *(dependency of `compute_table_values`)*
- **`analyze_cost_gain.py`** — misspecification cost measured on gain-over-random.
- **`analyze_interaction.py`** — direct interaction test (does misspecification cost BALD more than US?).
- **`compute_elink_posthoc.py`** — post-hoc `E_link` recomputation/validation from saved samples.

### Estimation-error diagnostic
- **`analyze_est_correct_link.py`** — correct-link gate analysis: `epsilon^Lap` by arm plus
  the FTRL active-over-random gain (`tab:est_gate`).
- **`measure_est_discrepancy.py`** — core: discrepancy between the Laplace–Taylor acquisition
  score and an importance-sampling reference on the same FTRL posterior.
- **`run_est_discrepancy_correct.py`** — drives `measure_est_discrepancy` under correctly
  specified links; produces `tab:est_gap` (n=536 Spearman). *(imports `measure_est_discrepancy`)*

### Data generation / extension runners
- **`run_ftrlbt_diag.py`** — FTRL-BT diagnostics (`epsilon^Lap`) on `exp_dataset` (BT, 0%).
- **`run_ftrlbt_inc_diag.py`** — FTRL-BT diagnostics on `exp_dataset_inc` (BT, 20%).
- **`run_estdiag_metrics.py`** — recompute RAI/ASRS for the `*_estdiag` arms.
- **`run_extend_hms.py`** — extend the BAYES arms to the per-dataset ceilings (link tables).
- **`run_extend_metrics.py`** — recompute RAI/ASRS for the extended arms.
