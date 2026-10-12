"""
Novelty 1: compare baselines and ML models with a time-based split.

    python compare_models.py
    python compare_models.py --horizons 1 6 24 48 --models ridge hgb lightgbm linearsvr
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from features import FEATURES, FORECAST_TARGETS, LOCKDOWN_START, load_dataset, split_masks
from modeling import make_model, regression_metrics

RESULTS = 'results'


def add_rows(rows, model, target, h, te, pred):
    y = te[f'y_{target}_h{h}']
    pred = pd.Series(np.asarray(pred, dtype=float), index=te.index)
    pre = te['Datetime'] < LOCKDOWN_START
    segments = {'test_all': pd.Series(True, index=te.index),
                'test_pre_lockdown': pre, 'test_lockdown': ~pre}
    for seg, mask in segments.items():
        ok = mask & pred.notna() & y.notna()
        if ok.sum() < 50:
            continue
        rows.append({'model': model, 'target': target, 'horizon': h, 'segment': seg,
                     **regression_metrics(y[ok], pred[ok])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizons', nargs='+', type=int, default=[1, 24])
    ap.add_argument('--targets', nargs='+', default=FORECAST_TARGETS)
    ap.add_argument('--models', nargs='+', default=['ridge', 'hgb', 'lightgbm'])
    ap.add_argument('--max-train-rows', type=int, default=250_000)
    ap.add_argument('--svr-rows', type=int, default=60_000)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    data = load_dataset()
    train_m, _, test_m = split_masks(data)
    print(f"train origins: {train_m.sum():,}  test origins: {test_m.sum():,}")

    rows = []
    t0 = time.time()
    for target in args.targets:
        for h in args.horizons:
            ycol = f'y_{target}_h{h}'
            tr = data[train_m & data[ycol].notna()]
            te = data[test_m & data[ycol].notna()]
            if len(tr) < 1000 or len(te) < 100:
                print(f"skip {ycol}: too few rows")
                continue
            print(f"\n=== {target}, +{h} h  (train {len(tr):,} / test {len(te):,}) ===")

            # Baselines
            add_rows(rows, 'persistence', target, h, te, te[f'cur_{target}'])
            station_mean = tr.groupby('StationId', observed=True)[ycol].mean().to_dict()
            add_rows(rows, 'station_mean', target, h, te,
                     te['StationId'].astype(str).map(station_mean).astype(float).fillna(tr[ycol].mean()))

            # ML models
            for name in args.models:
                n = args.svr_rows if name == 'linearsvr' else args.max_train_rows
                sub = tr.sample(min(len(tr), n), random_state=0)
                model = make_model(name)
                s = time.time()
                model.fit(sub[FEATURES], sub[ycol])
                pred = model.predict(te[FEATURES])
                add_rows(rows, name, target, h, te, pred)
                print(f"  {name:<10} fitted in {time.time() - s:5.1f}s")

    res = pd.DataFrame(rows)
    base = (res[res.model == 'persistence'][['target', 'horizon', 'segment', 'MAE']]
            .rename(columns={'MAE': 'MAE_persistence'}))
    res = res.merge(base, on=['target', 'horizon', 'segment'], how='left')
    res['MAE_skill_vs_persistence'] = 1 - res['MAE'] / res['MAE_persistence']
    res = res.drop(columns='MAE_persistence').round(4)
    res.to_csv(os.path.join(RESULTS, 'model_comparison.csv'), index=False)

    # Best model per (target, horizon) on the full test set
    full = res[res.segment == 'test_all']
    best = (full.loc[full.groupby(['target', 'horizon'])['MAE'].idxmin()]
            [['target', 'horizon', 'model', 'MAE', 'R2', 'MAE_skill_vs_persistence']])
    best.to_json(os.path.join(RESULTS, 'best_models.json'), orient='records', indent=2)

    print(f"\nTotal {time.time() - t0:.0f}s. Best models (test_all):")
    print(best.to_string(index=False))


if __name__ == '__main__':
    main()
