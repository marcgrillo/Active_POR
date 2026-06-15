"""
Illustrative case study: 20 European countries x 4 higher-education criteria.
Simulated DM answers each pairwise query deterministically by the listed (reference)
ranking -- Germany best ... Slovakia worst (0% inconsistency oracle; true_utility=None).
Runs the four POR variants with the BALD+US acquisition to a fixed budget T and saves the
per-step value-function state {j}_active.npy used to build the utility-evolution figure
and the final-RAI table.
"""
import os
import argparse
import numpy as np
from experiments.simulation import process_single_table

# --- performance matrix from table.tex (listed best -> worst) ---
NAMES = ['GER','UK','SPA','SWE','NET','ITA','FIN','BEL','AUS','DEN',
         'FRA','CZE','POR','SLO','IRE','HUN','EST','GRE','POL','SVK']
TABLE = np.array([
    [82,94,80,91],[74,91,96,82],[59,73,72,67],[47,77,90,46],[50,73,88,47],
    [51,50,84,55],[42,59,88,39],[44,57,84,41],[42,53,88,38],[42,61,68,39],
    [45,37,80,44],[41,43,80,40],[41,41,60,40],[38,37,72,34],[40,40,60,34],
    [39,34,48,38],[38,36,44,34],[39,28,40,34],[39,26,36,37],[37,21,8,37],
], dtype=float)

VARIANTS = ['BAYES-LIN', 'BAYES-BT', 'FTRL-LIN', 'FTRL-BT']


def build_ground_truth(n, seed=0):
    """true_ranking = listed order (rank i = i). ground_truth_prefs = all C(n,2) pairs
    oriented winner-first by that ranking, shuffled to serve as the random query sequence
    and the consistency lookup."""
    true_ranking = np.arange(n)                      # rank[i]=i  (0 = best)
    pairs = [[i, j] for i in range(n) for j in range(i + 1, n)]  # i<j => i better
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    return true_ranking, [np.array(p) for p in pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=15)
    ap.add_argument('--variants', nargs='+', default=VARIANTS)
    ap.add_argument('--methods', nargs='+', default=['BALD', 'US'])
    ap.add_argument('--outdir', default=os.path.join('scratch', 'case_study'))
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()

    n = len(TABLE)
    true_ranking, gt_prefs = build_ground_truth(n)
    os.makedirs(a.outdir, exist_ok=True)
    np.save(os.path.join(a.outdir, '_table.npy'), TABLE)

    for alg in a.variants:
        for method in a.methods:
            out = os.path.join(a.outdir, f'{alg}_{method}')   # one dir per (variant, acquisition)
            os.makedirs(out, exist_ok=True)
            print(f'\n=== {alg} | {method} | T={a.T} ===')
            process_single_table(
                table=TABLE,
                ground_truth_prefs=gt_prefs,
                true_ranking=true_ranking,
                true_utility=None,                 # deterministic oracle from the ranking
                num_steps=a.T,
                output_dir=out,
                alg=alg,
                active_method=method,
                overwrite=a.overwrite,
                n_samples_mc=2000,
                use_is_bald=True,
                use_mh_sampler=True,
                use_hmc_sampler=True,
                n_samples_mcmc=2000,
                compute_error_diagnostics=False,
                disable_tqdm=False,
            )
    print('\nDONE case-study runs.')


def _ensure(d, f):
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f)


if __name__ == '__main__':
    main()
