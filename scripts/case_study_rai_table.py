"""Emit the case-study RAI table in the table_ex.tex format (countries x methods x ranks,
modal value bolded), for the BT learners at budget T, under BALD and US."""
import numpy as np, os
from mcda.models import PiecewiseLinearTransformer
from case_study_run import TABLE, NAMES

T = 12
tr = PiecewiseLinearTransformer.from_equal_intervals(TABLE, 3)
X = tr.transform(TABLE); n = len(TABLE)
FULL = {'GER': 'Germany', 'UK': 'United Kingdom', 'SPA': 'Spain', 'SWE': 'Sweden',
        'NET': 'Netherlands', 'ITA': 'Italy'}
METHODS = [('BAYES-BT (BALD)', 'BAYES-BT_BALD', True),
           ('BAYES-BT (US)', 'BAYES-BT_US', True),
           ('BAYES-BT (random)', 'BAYES-BT_BALD', False)]    # passive track = random querying
ROWS = ['GER', 'UK', 'SPA', 'SWE', 'NET', 'ITA']           # reference listed order


def rai_pct(d, active):
    f = f'{T}_active.npy' if active else f'{T}.npy'
    w = np.load(os.path.join('scratch/case_study', d, f))
    W = np.atleast_2d(w); U = X @ W.T; rc = np.zeros((n, n))
    for s in range(U.shape[1]):
        rc[np.arange(n), np.argsort(np.argsort(-U[:, s]))] += 1
    return 100 * rc / U.shape[1]


R = {lab: rai_pct(d, act) for lab, d, act in METHODS}
is_ftrl = {lab: lab.startswith('FTRL') for lab, *_ in METHODS}
idx = {sn: k for k, sn in enumerate(NAMES)}

# rank columns to show: up to the last rank carrying >=0.5% for any shown (country, method)
maxr = 0
for sn in ROWS:
    i = idx[sn]
    for lab, *_ in METHODS:
        nz = np.where(R[lab][i] > 0.5)[0]
        if len(nz):
            maxr = max(maxr, nz.max())
ranks = list(range(maxr + 1))


def fmt(lab, i, r):
    v = R[lab][i][r]
    modal = (r == int(np.argmax(R[lab][i])))
    s = f'{int(round(v))}' if is_ftrl[lab] else f'{v:.1f}'
    return f'\\textbf{{ {s} }}' if modal else s


def cell(i, r):
    if all(R[lab][i][r] < 0.05 for lab, *_ in METHODS):
        return ''                                            # empty column for this country
    body = ' \\\\ '.join(fmt(lab, i, r) for lab, *_ in METHODS)
    return f'\\begin{{tabular}}{{c}} {body} \\end{{tabular}}'


colspec = 'cc|' + 'c' * len(ranks)
header = 'Country & Method & ' + ' & '.join(str(r + 1) for r in ranks) + ' \\\\'
lines = [
    '\\begin{table*}[ht]', '    \\centering',
    '    \\caption{Rank acceptability indices (in \\%) at budget $T=12$ for the six '
    'alternatives heading the FTRL-BT recommendation, under the BALD and US acquisitions; '
    'the modal rank of each method is in bold.}',
    '    \\resizebox{\\textwidth}{!}{',
    f'    \\begin{{tabular}}{{{colspec}}}', '    \\toprule', header, '\\midrule']
for sn in ROWS:
    i = idx[sn]
    mblock = '\\begin{tabular}{r} ' + ' '.join(f'{lab} \\\\ ' for lab, *_ in METHODS) + '\\end{tabular}'
    cells = ' & '.join(cell(i, r) for r in ranks)
    lines.append(f'{FULL[sn]} & {mblock} & {cells} \\\\ \\hline')
lines[-1] = lines[-1].replace(' \\hline', '')               # drop last hline before bottomrule
lines += ['\\bottomrule', '    \\end{tabular}', '    }', '    \\label{tab:final_rais}', '\\end{table*}']
out = '\n'.join(lines)
open('case_study_rai_table_out.tex', 'w', encoding='utf-8').write(out)
print(out)
