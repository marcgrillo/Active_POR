"""
Direct interaction test (the rigorous quantity): within each dataset, does link
misspecification cost BALD more than US? Active AULC, paired by table, t<=TMAX.
correct/misspec models per dataset:
  logit-20% (exp_dataset_inc): correct=BT, misspec=LIN
  linear    (lin_dataset_pwl): correct=LIN, misspec=BT
"""
import argparse
import numpy as np
from scipy.stats import wilcoxon
from experiments.aggregate import load_curve

F3 = 100; F2 = [2, 6, 10]
DS = [('exp_dataset_inc', 'logit-20%', 'BT', 'LIN'),   # ds, label, correct, misspec
      ('lin_dataset_pwl', 'linear', 'LIN', 'BT')]


def aulc(ds, sf, f1, f2, tmax):
    ndm = int(round(F3 * f1 * (f1 - 1) / 200))
    c = load_curve('asrs', ds, sf, f1, f2, F3, ndm, 'active')
    return np.nanmean(c[:min(tmax, c.shape[0])], axis=0) if c.size else None


def paired_cost(ds, method, correct, misspec, f1, tmax):
    """active AULC: correct-model minus misspec-model, paired by table, pooled F2."""
    d = []
    for f2 in F2:
        a = aulc(ds, f'BAYES-{correct}_{method}', f1, f2, tmax)
        b = aulc(ds, f'BAYES-{misspec}_{method}', f1, f2, tmax)
        if a is None or b is None:
            continue
        n = min(len(a), len(b)); d += (a[:n] - b[:n]).tolist()
    return np.array([x for x in d if np.isfinite(x)])


def rep(name, v):
    m = v.mean(); se = 2 * v.std(ddof=1) / np.sqrt(len(v)); p = wilcoxon(v).pvalue
    print(f'  {name:38} {m:+.4f} +- {se:.4f}  p={p:.1e}{"*" if p<0.05 else " "} (n={len(v)})')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--f1', type=int, default=30)
    ap.add_argument('--tmax', type=int, default=50); a = ap.parse_args()
    for ds, lab, cor, mis in DS:
        print(f'\n=== {lab}  (correct={cor}, misspec={mis})  t<={a.tmax} ===')
        us = paired_cost(ds, 'US', cor, mis, a.f1, a.tmax)
        ba = paired_cost(ds, 'BALD', cor, mis, a.f1, a.tmax)
        rep('US   misspec cost (correct-misspec)', us)
        rep('BALD misspec cost (correct-misspec)', ba)
        n = min(len(us), len(ba))
        rep('interaction (US cost - BALD cost)', us[:n] - ba[:n])


if __name__ == '__main__':
    main()
