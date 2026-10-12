"""
Shared feature engineering for forecasting.

Build and cache the dataset:   python features.py --sample-frac 0.15
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

DATA_PATH = os.path.join('data', 'station_hour_cleaned.csv')
DATASET_PATH = os.path.join('cache', 'forecast_dataset.pkl')

ALL_POLLUTANTS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
                  'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
FORECAST_TARGETS = ['AQI', 'PM2.5', 'PM10', 'NO2', 'O3']
HORIZONS = [1, 6, 24, 48]
LAGS = (6, 24)

# Time-based split (origin times)
GAP = pd.Timedelta(hours=48)
TRAIN_END = pd.Timestamp('2019-01-01')
VAL_END = pd.Timestamp('2020-01-01')
LOCKDOWN_START = pd.Timestamp('2020-03-25')


def feature_columns():
    cols = ['Hour', 'Month', 'DayOfWeek', 'doy_sin', 'doy_cos']
    for p in ALL_POLLUTANTS:
        cols += [f'cur_{p}', f'lag6_{p}', f'lag24_{p}', f'roll24_{p}']
    return cols


FEATURES = ['StationId'] + feature_columns()


def load_hourly(path=DATA_PATH):
    df = pd.read_csv(path, usecols=['StationId', 'Datetime'] + ALL_POLLUTANTS, low_memory=False)
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True, errors='coerce').dt.tz_localize(None)
    return df.dropna(subset=['Datetime'])


def build_station_frame(g, horizons=HORIZONS, targets=FORECAST_TARGETS):
    """
    g: DataFrame with a 'Datetime' column and the 13 pollutant columns (one station).
    Returns a frame indexed by every hour between min and max Datetime, with
    features at that hour and (optionally) y_{target}_h{h} = value at +h hours.
    The same function is used at training time and inside the web app, so there is
    no train/serve skew.
    """
    g = g.drop_duplicates('Datetime').set_index('Datetime').sort_index()
    idx = pd.date_range(g.index.min(), g.index.max(), freq='h')
    g = g.reindex(idx)

    cols = {}
    for p in ALL_POLLUTANTS:
        s = g[p].astype('float32')
        cols[f'cur_{p}'] = s
        for lag in LAGS:
            cols[f'lag{lag}_{p}'] = s.shift(lag)
        cols[f'roll24_{p}'] = s.rolling(24, min_periods=12).mean()
    for t in targets:
        for h in horizons:
            cols[f'y_{t}_h{h}'] = g[t].astype('float32').shift(-h)

    out = pd.DataFrame(cols, index=idx)
    out['Hour'] = idx.hour
    out['Month'] = idx.month
    out['DayOfWeek'] = idx.dayofweek
    doy = idx.dayofyear.values
    out['doy_sin'] = np.sin(2 * np.pi * doy / 366)
    out['doy_cos'] = np.cos(2 * np.pi * doy / 366)
    out.index.name = 'Datetime'
    return out


def build_dataset(sample_frac=0.15, seed=42, horizons=HORIZONS, targets=FORECAST_TARGETS):
    start = time.time()
    rng = np.random.default_rng(seed)
    df = load_hourly()
    print(f"Loaded {len(df):,} rows in {time.time() - start:.0f}s")

    parts = []
    for sid, g in df.groupby('StationId'):
        if len(g) < 200:
            continue
        f = build_station_frame(g, horizons, targets)
        ycols = [c for c in f.columns if c.startswith('y_')]
        f = f[f['cur_AQI'].notna() | f['cur_PM2.5'].notna()]
        f = f.dropna(subset=ycols, how='all')
        if sample_frac < 1.0:
            f = f.sample(frac=sample_frac, random_state=int(rng.integers(0, 10**9)))
        f.insert(0, 'StationId', sid)
        parts.append(f.reset_index())

    data = pd.concat(parts, ignore_index=True)
    station_ids = sorted(data['StationId'].unique())
    data['StationId'] = pd.Categorical(data['StationId'], categories=station_ids)

    os.makedirs('cache', exist_ok=True)
    data.to_pickle(DATASET_PATH)
    print(f"Dataset: {len(data):,} rows, {len(station_ids)} stations, "
          f"{data['Datetime'].min()} -> {data['Datetime'].max()} ({time.time() - start:.0f}s)")
    return data


def load_dataset(path=DATASET_PATH):
    if not os.path.exists(path):
        print("Dataset cache missing, building it now...")
        return build_dataset()
    return pd.read_pickle(path)


def split_masks(data):
    t = data['Datetime']
    train = t < (TRAIN_END - GAP)
    val = (t >= TRAIN_END) & (t < (VAL_END - GAP))
    test = t >= VAL_END
    return train, val, test


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-frac', type=float, default=0.15,
                    help='fraction of rows kept per station (1.0 = everything, needs lots of RAM)')
    build_dataset(ap.parse_args().sample_frac)
