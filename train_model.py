"""
Train multi-output regression for air-quality pollutants.

Usage:
    python train_model.py                          # Ridge, ~4% sample
    python train_model.py --sample-frac 0.2        # Ridge, 20% sample
    python train_model.py --model linearsvr        # scaled LinearSVR
"""
import argparse
import json
import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('train_model.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

DATA_PATH = os.path.join('data', 'station_hour_cleaned.csv')
TARGETS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
           'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
LAGS = [f'{t}_lag1' for t in TARGETS]
TIME_FEATURES = ['Year', 'Month', 'Day', 'Hour']
FEATURES = ['StationId'] + TIME_FEATURES + LAGS

MODEL_FILES = {
    'ridge':     ('Ridge',     'models/air_quality_model.pkl',     'models/accuracy_metrics.json'),
    'linearsvr': ('LinearSVR', 'models/air_quality_model_svm.pkl', 'models/accuracy_metrics_svm.json'),
}


def load_and_build_features(path):
    """Load full data, build lag features on the COMPLETE sorted series."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, usecols=['StationId', 'Datetime'] + TARGETS, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows")

    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    bad = df['Datetime'].isna().sum()
    if bad:
        logger.warning(f"Dropping {bad} rows with unparseable Datetime")
        df = df.dropna(subset=['Datetime'])

    df = df.sort_values(['StationId', 'Datetime']).reset_index(drop=True)

    means = df[TARGETS].mean()
    grouped = df.groupby('StationId')
    for t in TARGETS:
        df[f'{t}_lag1'] = grouped[t].shift(1).fillna(means[t])

    df['Year'] = df['Datetime'].dt.year
    df['Month'] = df['Datetime'].dt.month
    df['Day'] = df['Datetime'].dt.day
    df['Hour'] = df['Datetime'].dt.hour
    return df.drop(columns=['Datetime'])


def clean_numeric(data):
    """Median-fill NaNs and clip targets/lags to the 1st-99th percentile. Returns bounds for lags."""
    bounds = {}
    for col in TARGETS + LAGS:
        data[col] = data[col].fillna(data[col].median())
        low, high = data[col].quantile([0.01, 0.99])
        data[col] = data[col].clip(low, high)
        if col in LAGS:
            bounds[col] = [float(low), float(high)]
    return data, bounds


def build_model(kind):
    cat = ('cat', OneHotEncoder(handle_unknown='ignore'), ['StationId'])
    if kind == 'ridge':
        pre = ColumnTransformer([cat], remainder='passthrough')
        reg = Ridge(alpha=1.0)
    else:
        num = ('num', StandardScaler(), TIME_FEATURES + LAGS)
        pre = ColumnTransformer([cat, num])
        reg = MultiOutputRegressor(
            TransformedTargetRegressor(
                regressor=LinearSVR(epsilon=0.1, C=1.0, dual='auto', max_iter=5000, random_state=42),
                transformer=StandardScaler(),
            )
        )
    return Pipeline([('preprocessor', pre), ('regressor', reg)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODEL_FILES), default='ridge')
    parser.add_argument('--sample-frac', type=float, default=0.0386,
                        help='fraction of rows used for training (1.0 = all)')
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    name, model_path, metrics_path = MODEL_FILES[args.model]
    os.makedirs('models', exist_ok=True)

    data = load_and_build_features(DATA_PATH)
    if args.sample_frac < 1.0:
        data = data.sample(frac=args.sample_frac, random_state=42)
        logger.info(f"Sampled {len(data):,} rows ({args.sample_frac * 100:.2f}%)")

    data, bounds = clean_numeric(data)
    X, y = data[FEATURES], data[TARGETS]

    model = build_model(args.model)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    scores = {t: [] for t in TARGETS}

    start = time.time()
    for fold, (tr, te) in enumerate(kf.split(X), 1):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        for i, t in enumerate(TARGETS):
            scores[t].append(r2_score(y.iloc[te][t], pred[:, i]))
        logger.info(f"Fold {fold}/{args.folds} done ({time.time() - start:.0f}s)")

    mean_r2 = {t: float(np.mean(v)) for t, v in scores.items()}
    logger.info(f"Mean R2: {mean_r2}")

    logger.info("Fitting final model on all sampled rows...")
    model.fit(X, y)
    joblib.dump(model, model_path)
    logger.info(f"Saved model -> {model_path}")

    with open(metrics_path, 'w') as f:
        json.dump({'model': name, 'n_rows': int(len(data)), 'cv_folds': args.folds,
                   'sample_frac': args.sample_frac, 'r2': mean_r2}, f, indent=4)
    logger.info(f"Saved metrics -> {metrics_path}")

    if args.model == 'ridge':   # the app uses the Ridge model's bounds
        with open('models/clip_bounds.json', 'w') as f:
            json.dump(bounds, f, indent=2)
        logger.info("Saved clip bounds -> models/clip_bounds.json")


if __name__ == '__main__':
    main()