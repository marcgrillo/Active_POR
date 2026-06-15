"""
Misspecification cost measured the only valid way: on the GAIN over each model's own
random baseline (ASRS level depends on inconsistency, so raw active AULC is confounded).

per (dataset, method): cost = gain_correct - gain_misspec, paired by table, where
gain = AULC(active) - AULC(passive). Then interaction = US cost - BALD cost.
  logit-20% (exp_dataset_inc): correct=BT, misspec=LIN
  linear    (lin_dataset_pwl): correct=LIN, misspec=BT
"""
import argparse
import numpy as np
from scipy.stats import wilcoxon
from experiments.aggregate import load_curve

F3 = 100; F2 = [2, 6, 10]
DS = [('exp_dataset_inc', 'logit-20%', 'BT', 'LIN'),
      ('lin_dataset_pwl', 'linear', 'LIN', 'BT')]


def gain(ds, model, method, f1, f2, tmax):
    """Per-HM gain over random = AULC(active) - AULC(passive), over t<=tmax."""
    ndm = int(round(F3 * f1 * (f1 - 1) / 200))
    sf = f'BAYES-{model}_{method}'
    act = load_curve('asrs', ds, sf, f1, f2, F3, ndm, 'active')
    pas = load_curve('asrs', ds, sf, f1, f2, F3, ndm, 'passive')
    if not act.size or not pas.size:
        return None
    T = min(tmax, act.shape[0], pas.shape[0])
    a = np.nanmean(act[:T], axis=0); r = np.nanmean(pas[:T], axis=0)
    n = min(len(a), len(r))
    return a[:n] - r[:n]


def cost(ds, method, correct, misspec, f1, tmax):
    """gain_correct - gain_misspec, paired by table, pooled F2."""
    d = []
    for f2 in F2:
        gc = gain(ds, correct, method, f1, f2, tmax)
        gm = gain(ds, misspec, method, f1, f2, tmax)
        if gc is None or gm is None:
            continue
        n = min(len(gc), len(gm)); d += (gc[:n] - gm[:n]).tolist()
    return np.array([x for x in d if np.isfinite(x)])


def rep(name, v):
    m = v.mean(); se = 2 * v.std(ddof=1) / np.sqrt(len(v)); p = wilcoxon(v).pvalue
    print(f'  {name:42} {m:+.4f} +- {se:.4f}  p={p:.1e}{"*" if p<0.05 else " "} (n={len(v)})')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--f1', type=int, default=30)
    ap.add_argument('--tmax', type=int, default=50); a = ap.parse_args()
    for ds, lab, cor, mis in DS:
        print(f'\n=== {lab}  (correct={cor}, misspec={mis})  GAIN-over-random, t<={a.tmax} ===')
        us = cost(ds, 'US', cor, mis, a.f1, a.tmax)
        ba = cost(ds, 'BALD', cor, mis, a.f1, a.tmax)
        rep('US   misspec cost (gain_correct - gain_misspec)', us)
        rep('BALD misspec cost (gain_correct - gain_misspec)', ba)
        n = min(len(us), len(ba))
        rep('interaction (US cost - BALD cost)', us[:n] - ba[:n])


if __name__ == '__main__':
    main()
