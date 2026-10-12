"""
Global SHAP analysis for a LightGBM model.

    python explain.py --target AQI --horizon 24
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shap

from features import FEATURES, load_dataset, split_masks
from modeling import make_model

RESULTS = 'results'


def family(col):
    if col == 'StationId':
        return 'Station'
    if col in ('Hour', 'Month', 'DayOfWeek', 'doy_sin', 'doy_cos'):
        return 'Time / season'
    return {'cur': 'Current reading', 'lag6': 'Reading 6 h earlier',
            'lag24': 'Reading 24 h earlier', 'roll24': '24 h average'}[col.split('_')[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='AQI')
    ap.add_argument('--horizon', type=int, default=24)
    ap.add_argument('--n-explain', type=int, default=2000)
    ap.add_argument('--max-train-rows', type=int, default=250_000)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    ycol = f'y_{args.target}_h{args.horizon}'
    data = load_dataset()
    train_m, _, test_m = split_masks(data)
    tr = data[train_m & data[ycol].notna()]
    tr = tr.sample(min(len(tr), args.max_train_rows), random_state=0)
    te = data[test_m & data[ycol].notna()]
    te = te.sample(min(len(te), args.n_explain), random_state=0)

    model = make_model('lightgbm')
    model.fit(tr[FEATURES], tr[ycol])

    # LightGBM's own TreeSHAP: last column is the expected value
    contrib = model.predict(te[FEATURES], pred_contrib=True)
    sv = contrib[:, :-1]
    importance = np.abs(sv).mean(axis=0)

    by_feature = sorted(zip(FEATURES, importance), key=lambda x: -x[1])
    fam = {}
    for col, imp in by_feature:
        fam[family(col)] = fam.get(family(col), 0.0) + float(imp)

    tag = f'{args.target}_h{args.horizon}'
    with open(os.path.join(RESULTS, f'shap_{tag}.json'), 'w') as f:
        json.dump({'target': args.target, 'horizon': args.horizon,
                   'top_features': [{'feature': c, 'mean_abs_shap': float(v)} for c, v in by_feature[:20]],
                   'by_family': dict(sorted(fam.items(), key=lambda x: -x[1]))}, f, indent=2)

    Xp = te[FEATURES].copy()
    Xp['StationId'] = Xp['StationId'].cat.codes
    plt.figure()
    shap.summary_plot(sv, Xp, max_display=15, show=False)
    plt.title(f'SHAP: {args.target} at +{args.horizon} h')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f'shap_{tag}.png'), dpi=140)
    plt.close()
    print(f"Saved results/shap_{tag}.png and .json")
    print(json.dumps(dict(sorted(fam.items(), key=lambda x: -x[1])), indent=2))


if __name__ == '__main__':
    main()
