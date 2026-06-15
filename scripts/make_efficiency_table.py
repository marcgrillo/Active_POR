"""
Efficiency table: eta(alpha) = ASRS retained / budget used, for each alpha, all four
arms, both inconsistency regimes. The raw ratio is monotone increasing in alpha (stopping
earlier always raises retained/budget), so its unconstrained maximum is degenerate
(alpha->1, negligible retention). We therefore highlight, per (arm x regime) column, the
alpha that maximizes eta subject to retaining at least RET_FLOOR of the final ASRS -- the
most economical stop that still preserves the bulk of attainable quality.
"""
import pickle
import numpy as np
from stopping_eval import pareto_rule_relative, REGIMES, ALGOS, RAW

GRID = [0.001, 0.01, 0.05, 0.10, 0.25, 0.45, 0.65, 0.85, 0.95]
RET_FLOOR = 0.90


def main():
    raw = pickle.load(open(RAW, 'rb'))
    a = np.array(GRID)
    # data[(ds,engine,method)] = (ret, bud, eff)
    data = {}
    for engine, method, _, _ in ALGOS:
        for ds, _ in REGIMES:
            bud, ret, _ = pareto_rule_relative(raw[(ds, engine, method)], a)
            eff = ret / np.clip(bud, 1e-6, None)
            data[(ds, engine, method)] = (ret, bud, eff)

    # highlighted row index per column = argmax eff s.t. ret >= floor
    hi = {}
    for key, (ret, bud, eff) in data.items():
        ok = np.where(ret >= RET_FLOOR)[0]
        # best eff that preserves >= floor; if floor unreachable, the highest-retention stop
        hi[key] = int(ok[np.argmax(eff[ok])]) if len(ok) else int(np.argmax(ret))

    head = ' & '.join(r'\textbf{%s-%s}' % (e, m) for e, m, _, _ in ALGOS)
    L = []
    L.append(r'\begin{table*}[t]\centering\small')
    L.append(r'\caption{Elicitation efficiency $\eta(\alpha)=\mathrm{ASRS\ retained}/'
             r'\mathrm{budget\ used}$ of the stopping rule, for every threshold $\alpha$, '
             r'all four learners and both inconsistency regimes ($F_1=30$, pooled over '
             r'$F_2\in\{2,6,10\}$ and human models). Larger is better. Because $\eta$ '
             r'increases monotonically as $\alpha\!\to\!1$ (stopping earlier mechanically '
             r'raises the ratio but discards quality), we highlight in \textbf{bold}, for '
             r'each learner and regime, the $\alpha$ that maximizes $\eta$ while retaining '
             r'at least $90\%$ of the final ASRS, i.e.\ the most economical stop that '
             r'preserves quality.}')
    L.append(r'\label{tab:stopping_eff}')
    L.append(r'\setlength{\tabcolsep}{5pt}')
    L.append(r'\begin{tabular}{l cccc c cccc}')
    L.append(r'\toprule')
    L.append(r' & \multicolumn{4}{c}{$0\%$ inconsistency} & & '
             r'\multicolumn{4}{c}{$20\%$ inconsistency} \\')
    L.append(r'\cmidrule(lr){2-5}\cmidrule(lr){7-10}')
    L.append(r'$\alpha$ & ' + head + ' & & ' + head + r' \\')
    L.append(r'\midrule')
    for r, al in enumerate(GRID):
        cells = []
        for ds, _ in REGIMES:
            for e, m, _, _ in ALGOS:
                eff = data[(ds, e, m)][2][r]
                s = f'{eff:.1f}'
                if hi[(ds, e, m)] == r:
                    s = r'\textbf{' + s + '}'
                cells.append(s)
        L.append(f'{al:g} & ' + ' & '.join(cells[:4]) + ' & & ' + ' & '.join(cells[4:]) + r' \\')
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\end{table*}')
    print('\n'.join(L))

    # also print the highlighted (alpha, ret, bud) for the write-up
    print('\n% highlighted operating points (ret>=0.9):')
    for (ds, e, m), idx in hi.items():
        ret, bud, eff = data[(ds, e, m)]
        print(f'%   {ds:16} {e}-{m:5}: alpha={GRID[idx]:.2f}  '
              f'ret={ret[idx]*100:.0f}%  bud={bud[idx]*100:.0f}%  eta={eff[idx]:.1f}')


if __name__ == '__main__':
    main()