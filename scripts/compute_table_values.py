"""
All numbers (mean, 2*SEM, Wilcoxon p) for the link-error subsection tables.
Table A: LIN model, US vs BALD, on 0%-inconsistency (misspec) vs lin (correct) data:
         normalized gain over random, BALD-US contrast, and E_link.
Table B: BT vs LIN on logit-20% data: normalized misspecification cost + interaction.
"""
import numpy as np
from scipy.stats import wilcoxon
from experiments.aggregate import load_curve
from plot_h2 import collect

F3 = 100; f1 = 30; F2 = [2, 6, 10]; ndm = int(round(F3 * f1 * (f1 - 1) / 200))
CEIL = {'exp_dataset': 0.6683, 'lin_dataset_pwl': 0.3047, 'exp_dataset_inc': 0.4516}


def stat(v):
    v = np.array([x for x in v if np.isfinite(x)])
    return v.mean(), 2 * v.std(ddof=1) / np.sqrt(len(v)), wilcoxon(v).pvalue, len(v)


def naulc(ds, sf, tr, tmax):
    out = []
    for f2 in F2:
        c = load_curve('asrs', ds, sf, f1, f2, F3, ndm, tr)
        if c.size:
            out.append(np.nanmean(c[:min(tmax, c.shape[0])], axis=0))
    return out


def ngain(ds, sf, tmax):
    g = []
    for f2i in range(len(F2)):
        a = naulc(ds, sf, 'active', tmax)[f2i]; p = naulc(ds, sf, 'passive', tmax)[f2i]
        n = min(len(a), len(p)); g += ((a[:n] - p[:n]) / CEIL[ds]).tolist()
    return np.array(g)


def bald_minus_us(ds, tmax):
    d = []
    for f2i in range(len(F2)):
        b = naulc(ds, f'BAYES-LIN_BALD', 'active', tmax)[f2i]
        u = naulc(ds, f'BAYES-LIN_US', 'active', tmax)[f2i]
        n = min(len(b), len(u)); d += ((b[:n] - u[:n]) / CEIL[ds]).tolist()
    return np.array(d)


print('===================== TABLE A (LIN; full horizon) =====================')
elink = collect([30]); elink = elink[(elink.alg == 'BAYES') & (elink.f1 == 30)]
emap = {'exp_dataset': 'misspec', 'lin_dataset_pwl': 'correct'}
for ds, lab in [('exp_dataset', '0% inconsistency (misspec)'),
                ('lin_dataset_pwl', 'lin inconsistency (correct)')]:
    print(f'\n  {lab}:')
    for method in ['US', 'BALD']:
        m, se, p, n = stat(ngain(ds, f'BAYES-LIN_{method}', 435))
        el = elink[(elink.dataset == emap[ds]) & (elink.method == method)].E_link.dropna()
        em, ese = el.mean(), 2 * el.std(ddof=1) / np.sqrt(len(el))
        print(f'    {method:4}: norm.gain={m:+.4f}+-{se:.4f} p={p:.1e}  |  E_link={em:.3f}+-{ese:.3f} (n={len(el)})')
    bm, bse, bp, bn = stat(bald_minus_us(ds, 435))
    print(f'    BALD-US: {bm:+.4f}+-{bse:.4f}  p={bp:.1e} (n={bn})')

print('\n===================== TABLE B (BT vs LIN, logit-20%; t<=50) =====================')
ds = 'exp_dataset_inc'
def cost(method):
    d = []
    for f2i in range(len(F2)):
        gc = ngain(ds, f'BAYES-BT_{method}', 50)        # correct=BT
        gm = ngain(ds, f'BAYES-LIN_{method}', 50)       # misspec=LIN
    # recompute paired properly:
    cc = ngain(ds, f'BAYES-BT_{method}', 50); mm = ngain(ds, f'BAYES-LIN_{method}', 50)
    n = min(len(cc), len(mm)); return cc[:n] - mm[:n]
us = cost('US'); ba = cost('BALD')
for nm, v in [('US misspec cost', us), ('BALD misspec cost', ba)]:
    m, se, p, n = stat(v); print(f'  {nm:20}: {m:+.4f}+-{se:.4f}  p={p:.1e} (n={n})')
n = min(len(us), len(ba)); m, se, p, _ = stat(us[:n] - ba[:n])
print(f'  {"interaction":20}: {m:+.4f}+-{se:.4f}  p={p:.1e}')
