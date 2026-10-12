"""
Novelty 2: train quantile models for several horizons and calibrate with conformal prediction.

    python train_forecast.py                       # all targets x horizons (~10-20 min)
    python train_forecast.py --targets AQI --horizons 1 24   # quick test
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from features import (FEATURES, FORECAST_TARGETS, HORIZONS, LOCKDOWN_START,
                      TRAIN_END, VAL_END, load_dataset, split_masks)
from modeling import conformal_qhat, make_model

QUANTILES = [0.1, 0.5, 0.9]
ALPHA = 0.2          # 1 - ALPHA = 80% nominal coverage
RESULTS = 'results'


def coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=FORECAST_TARGETS)
    ap.add_argument('--horizons', nargs='+', type=int, default=HORIZONS)
    ap.add_argument('--n-estimators', type=int, default=250)
    ap.add_argument('--max-train-rows', type=int, default=300_000)
    args = ap.parse_args()

    os.makedirs('models', exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    data = load_dataset()
    train_m, val_m, test_m = split_masks(data)

    models, conformal, rows = {}, {}, []
    t0 = time.time()
    for target in args.targets:
        for h in args.horizons:
            ycol = f'y_{target}_h{h}'
            ok = data[ycol].notna()
            tr = data[train_m & ok]
            va = data[val_m & ok]
            te = data[test_m & ok]
            if len(tr) < 1000 or len(va) < 200 or len(te) < 100:
                print(f"skip {ycol}: too few rows ({len(tr)}/{len(va)}/{len(te)})")
                continue
            tr = tr.sample(min(len(tr), args.max_train_rows), random_state=0)

            for q in QUANTILES:
                m = make_model('lightgbm', quantile=q, n_estimators=args.n_estimators)
                m.fit(tr[FEATURES], tr[ycol])
                models[f'{target}|{h}|{q}'] = m

            def predict(df):
                p = np.column_stack([models[f'{target}|{h}|{q}'].predict(df[FEATURES]) for q in QUANTILES])
                p = np.sort(p, axis=1)                    # fix quantile crossing
                return p[:, 0], np.maximum(p[:, 1], 0), p[:, 2]

            # Conformal calibration on validation
            lo_v, _, hi_v = predict(va)
            yv = va[ycol].values
            qhat = conformal_qhat(np.maximum(lo_v - yv, yv - hi_v), ALPHA)
            conformal[f'{target}|{h}'] = qhat

            # Evaluation on test
            lo_r, med, hi_r = predict(te)
            yt = te[ycol].values
            lo_c = np.minimum(np.maximum(0, lo_r - qhat), med)
            hi_c = np.maximum(hi_r + qhat, med)
            pre = (te['Datetime'] < LOCKDOWN_START).values
            for seg, m_ in {'test_all': np.ones(len(te), bool),
                            'test_pre_lockdown': pre, 'test_lockdown': ~pre}.items():
                if m_.sum() < 50:
                    continue
                rows.append({
                    'target': target, 'horizon': h, 'segment': seg, 'n': int(m_.sum()),
                    'nominal': 1 - ALPHA, 'qhat': round(qhat, 3),
                    'coverage_raw': coverage(yt[m_], lo_r[m_], hi_r[m_]),
                    'coverage_conformal': coverage(yt[m_], lo_c[m_], hi_c[m_]),
                    'width_raw': float(np.mean(hi_r[m_] - lo_r[m_])),
                    'width_conformal': float(np.mean(hi_c[m_] - lo_c[m_])),
                    'MAE_median': float(np.mean(np.abs(yt[m_] - med[m_]))),
                })
            r = [x for x in rows if x['target'] == target and x['horizon'] == h and x['segment'] == 'test_all']
            if r:
                print(f"{target:>6} +{h:>2}h  qhat={qhat:7.2f}  coverage raw={r[0]['coverage_raw']:.2f} "
                      f"conformal={r[0]['coverage_conformal']:.2f}  ({time.time() - t0:.0f}s)")

    if not models:
        raise SystemExit("No models were trained. Check the dataset.")

    joblib.dump({'models': models, 'conformal': conformal}, 'models/forecast_models.joblib', compress=3)
    meta = {
        'station_ids': data['StationId'].cat.categories.tolist(),
        'feature_cols': FEATURES,
        'targets': args.targets,
        'horizons': args.horizons,
        'quantiles': QUANTILES,
        'alpha': ALPHA,
        'interval_level': 1 - ALPHA,
        'splits': {'train_before': str(TRAIN_END - pd.Timedelta(hours=48)),
                   'validation': f'{TRAIN_END.date()} to {VAL_END.date()}',
                   'test_from': str(VAL_END.date())},
    }
    with open('models/forecast_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(RESULTS, 'interval_coverage.csv'), index=False)
    print(f"\nSaved models/forecast_models.joblib, models/forecast_meta.json, "
          f"results/interval_coverage.csv  (total {time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
