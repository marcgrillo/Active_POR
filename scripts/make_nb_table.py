"""
Small table of the NB-optimal threshold per learner/method/regime:
  alpha* = argmax_alpha [ (r(alpha)-r_0) - lambda*rho(alpha) ],  with the corresponding
  ASRS retained and budget used. Uses the cached KL signals; no recompute.
"""
import pickle
import numpy as np
from stopping_eval import REGIMES, ALGOS, RAW
from plot_eta_gain import curves, ALPHAS

LAM = 1.0


def opt(data):
    bud, ret, gain = curves(data, ALPHAS)
    nb = gain - LAM * bud
    k = int(np.argmax(nb))
    return ALPHAS[k], ret[k], bud[k]


def main():
    raw = pickle.load(open(RAW, 'rb'))
    L = [r'\begin{table}[t]\centering\small',
         r'\caption{Net-benefit--optimal stopping threshold '
         r'$\alpha^\star=\arg\max_\alpha\,[(r(\alpha)-r_0)-\lambda\rho(\alpha)]$ '
         r'($\lambda=1$), with the resulting ASRS retained and budget used, per learner '
         r'and inconsistency regime ($F_1=30$, pooled over $F_2\in\{2,6,10\}$ and human '
         r'models). For BAYES the optimum is essentially the same threshold in both '
         r'regimes; for FTRL it is lower and shifts with the noise level.}',
         r'\label{tab:stopping_nb}',
         r'\setlength{\tabcolsep}{6pt}',
         r'\begin{tabular}{ll ccc c ccc}',
         r'\toprule',
         r' & & \multicolumn{3}{c}{$0\%$ inconsistency} & & '
         r'\multicolumn{3}{c}{$20\%$ inconsistency} \\',
         r'\cmidrule(lr){3-5}\cmidrule(lr){7-9}',
         r'Learner & Method & $\alpha^\star$ & ASRS ret. & Budget & & '
         r'$\alpha^\star$ & ASRS ret. & Budget \\',
         r'\midrule']
    for engine, method, _, _ in ALGOS:
        a0, r0, b0 = opt(raw[(REGIMES[0][0], engine, method)])
        a1, r1, b1 = opt(raw[(REGIMES[1][0], engine, method)])
        L.append(f'{engine} & {method} & {a0:.2f} & {r0*100:.0f}\\% & {b0*100:.0f}\\% & & '
                 f'{a1:.2f} & {r1*100:.0f}\\% & {b1*100:.0f}\\% \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    print('\n'.join(L))


if __name__ == '__main__':
    main()