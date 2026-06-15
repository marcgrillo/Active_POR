"""Recompute RAI + ASRS for the extended-HM BAYES arms."""
from experiments.metrics import BenchmarkRunner

F2 = [2, 6, 10]
F3 = [100]
JOBS = [('exp_dataset_inc', 25), ('lin_dataset_pwl', 20)]
ARMS = ['BAYES-LIN_US', 'BAYES-LIN_BALD', 'BAYES-BT_US', 'BAYES-BT_BALD']
f1 = 30


def main():
    for ds, hm in JOBS:
        for sf in ARMS:
            for f2 in F2:
                ndm = 60          # analysis horizon is t<=50; only need the first 60 steps
                print(f'Metrics: {ds} f1={f1} f2={f2} {sf} (hm={hm})')
                try:
                    r = BenchmarkRunner(dataset_fold=ds, sub_fold=sf, num_subint=3,
                                        hm=hm, F1=[f1], F2=[f2], F3=F3, num_dm_dec=ndm)
                    r.compute_metrics('rai', force=False)   # per-table; new HMs filled in
                    r.compute_asrs(force=True)              # re-aggregate across all HMs
                except Exception as e:
                    print(f'  FAILED: {e}')


if __name__ == '__main__':
    main()
