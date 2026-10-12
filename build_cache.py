"""
One-off script: reads the big hourly CSV in chunks and writes small cache files.

Usage:  python build_cache.py
Output: cache/global_means.json
        cache/station_means.csv
        cache/station_hour_means.csv
        cache/station_daily.csv
        cache/hourly.db  (SQLite, indexed on StationId + Datetime)
"""
import json
import os
import sqlite3
import time

import numpy as np
import pandas as pd

SRC = os.path.join('data', 'station_hour_cleaned.csv')
OUT = 'cache'
TARGETS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
           'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
CHUNK = 250_000


def main():
    if not os.path.exists(SRC):
        raise FileNotFoundError(f"{SRC} not found. Run from the project root.")
    os.makedirs(OUT, exist_ok=True)

    db_path = os.path.join(OUT, 'hourly.db')
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode = WAL;")
    con.execute("PRAGMA synchronous = NORMAL;")

    daily_sum, daily_cnt = [], []   # keyed by (StationId, Date)
    hour_sum, hour_cnt = [], []     # keyed by (StationId, Hour)

    start = time.time()
    total = 0
    for chunk in pd.read_csv(SRC, usecols=['StationId', 'Datetime'] + TARGETS,
                             chunksize=CHUNK, low_memory=False):
        dt = pd.to_datetime(chunk['Datetime'], utc=True, errors='coerce')
        chunk = chunk[dt.notna()].copy()
        dt = dt[dt.notna()]
        chunk['Date'] = dt.dt.strftime('%Y-%m-%d')
        chunk['Hour'] = dt.dt.hour
        chunk['Datetime'] = dt.dt.strftime('%Y-%m-%d %H:00:00')

        g = chunk.groupby(['StationId', 'Date'])[TARGETS]
        daily_sum.append(g.sum())
        daily_cnt.append(g.count())

        g = chunk.groupby(['StationId', 'Hour'])[TARGETS]
        hour_sum.append(g.sum())
        hour_cnt.append(g.count())

        chunk[['StationId', 'Datetime'] + TARGETS].to_sql(
            'hourly', con, if_exists='append', index=False)

        total += len(chunk)
        print(f"  processed {total:,} rows ({time.time() - start:.0f}s)")

    print("Creating SQLite index...")
    con.execute('CREATE INDEX IF NOT EXISTS idx_station_dt ON hourly (StationId, Datetime)')
    con.commit()
    con.close()

    def combine(sums, cnts):
        s = pd.concat(sums).groupby(level=list(range(pd.concat(sums).index.nlevels))).sum()
        c = pd.concat(cnts).groupby(level=list(range(pd.concat(cnts).index.nlevels))).sum()
        return s, c

    d_s, d_c = combine(daily_sum, daily_cnt)
    h_s, h_c = combine(hour_sum, hour_cnt)

    # Daily means per station (for line charts)
    daily = (d_s / d_c.replace(0, np.nan)).reset_index()
    daily.to_csv(os.path.join(OUT, 'station_daily.csv'), index=False)

    # Per-station means (for bar/pie charts and lag fallback)
    st_s = d_s.groupby(level=0).sum()
    st_c = d_c.groupby(level=0).sum()
    station_means = (st_s / st_c.replace(0, np.nan)).reset_index()
    station_means.to_csv(os.path.join(OUT, 'station_means.csv'), index=False)

    # Per-station-per-hour-of-day means (better lag fallback)
    station_hour_means = (h_s / h_c.replace(0, np.nan)).reset_index()
    station_hour_means.to_csv(os.path.join(OUT, 'station_hour_means.csv'), index=False)

    # Global means (last-resort fallback)
    global_means = (d_s.sum() / d_c.sum().replace(0, np.nan)).to_dict()
    with open(os.path.join(OUT, 'global_means.json'), 'w') as f:
        json.dump({k: float(v) for k, v in global_means.items()}, f, indent=2)

    print(f"Done in {time.time() - start:.0f}s. Rows: {total:,}")


if __name__ == '__main__':
    main()
