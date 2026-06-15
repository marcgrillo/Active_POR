"""Compute RAI + ASRS for the FTRL-BT estdiag sub_folds on exp_dataset_inc."""
from experiments.metrics import BenchmarkRunner

F2 = [2, 6, 10]
F3 = [100]
f1 = 30
ARMS = ['FTRL-BT_US_estdiag', 'FTRL-BT_BALD_estdiag']
DS = 'exp_dataset_inc'
HM = 10
NDM = 60

if __name__ == '__main__':
    for sf in ARMS:
        for f2 in F2:
            print(f'Metrics: {DS} f1={f1} f2={f2} {sf}')
            try:
                r = BenchmarkRunner(dataset_fold=DS, sub_fold=sf, num_subint=3,
                                    hm=HM, F1=[f1], F2=[f2], F3=F3, num_dm_dec=NDM)
                r.compute_metrics('rai', force=False)
                r.compute_asrs(force=True)
            except Exception as e:
                print(f'  FAILED: {e}')
    print('DONE estdiag metrics.')
