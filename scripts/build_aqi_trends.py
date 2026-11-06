"""Ingestion and analysis pipeline for post-2020 CPCB AQI dataset.

Reads daily AQI data (2020 - Oct 2025), dedupes, loads into SQLite
table daily_aqi(date, city, aqi, category) in cache/daily_aqi.db,
computes fixed panel metrics for the 25 Mar - 31 May window,
and writes data/aqi_trends.json.
"""
import argparse
import json
import logging
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(BASE_DIR, 'cache', 'daily_aqi.db')
OUTPUT_JSON = os.path.join(BASE_DIR, 'data', 'aqi_trends.json')

SANITY_BENCHMARKS = {
    'panel_size': 58,
    'median_window_aqi': {
        2020: 89.4,
        2021: 114.9,
        2022: 138.5,
        2023: 108.7,
        2024: 111.0,
        2025: 103.2,
    },
    'median_pct_change': {
        2021: 31.9,
        2022: 58.5,
        2023: 26.4,
        2024: 30.5,
        2025: 17.9,
    },
}

NAMED_CITIES = ['Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Bengaluru', 'Hyderabad', 'Lucknow', 'Ahmedabad']
ANALYSIS_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def find_csv_path(user_path=None):
    if user_path and os.path.exists(user_path):
        return user_path
    candidates = [
        os.path.join(BASE_DIR, 'data', 'new', 'AQI_2020_to_2025_October.csv'),
        os.path.join(BASE_DIR, 'data', 'AQI_2020_to_2025_October.csv'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Could not find AQI_2020_to_2025_October.csv in data/new/ or data/.")


def load_and_clean_csv(csv_path):
    """Read CSV, parse dates with dayfirst=True, strip city names, and average duplicates."""
    logger.info(f"Loading raw CSV from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows.")

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df['City'] = df['City'].astype(str).str.strip()
    df['AQI Value'] = pd.to_numeric(df['AQI Value'], errors='coerce')
    df = df.dropna(subset=['AQI Value', 'City'])

    # Average duplicates on (Date, City)
    dup_count = df.duplicated(subset=['Date', 'City']).sum()
    logger.info(f"Found {dup_count} duplicate (Date, City) rows. Averaging AQI values...")

    # Groupby to average AQI Value and preserve Air Quality
    dedup = df.groupby(['Date', 'City'], as_index=False).agg({
        'AQI Value': 'mean',
        'Air Quality': 'first',
    })
    logger.info(f"Rows after deduplication: {len(dedup):,}")
    return dedup


def store_in_sqlite(df, db_path=DB_PATH):
    """Store cleaned records in a separate SQLite database daily_aqi(date, city, aqi, category)."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    logger.info(f"Storing daily_aqi in SQLite at {db_path}...")
    con = sqlite3.connect(db_path)
    try:
        sub = pd.DataFrame({
            'date': df['Date'].dt.strftime('%Y-%m-%d'),
            'city': df['City'],
            'aqi': df['AQI Value'].astype(float),
            'category': df['Air Quality'].astype(str)
        })
        sub.to_sql('daily_aqi', con, if_exists='replace', index=False)
        cur = con.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_aqi_city_date ON daily_aqi (city, date)")
        con.commit()
        logger.info(f"Stored {len(sub):,} records in daily_aqi table.")
    finally:
        con.close()


def compute_fixed_panel(df, years=ANALYSIS_YEARS, threshold_pct=0.80):
    """Compute fixed panel of cities having >= 80% of window days in every year 2020-2025."""
    m = df['Date'].dt.month
    d = df['Date'].dt.day
    window_mask = ((m == 3) & (d >= 25)) | (m == 4) | (m == 5)
    wdf = df[window_mask].copy()
    wdf['Year'] = wdf['Date'].dt.year

    total_window_days = 68  # 7 in March + 30 in April + 31 in May
    min_days_required = total_window_days * threshold_pct  # 54.4 days

    counts = wdf.groupby(['City', 'Year'])['AQI Value'].count().unstack(fill_value=0)
    panel = sorted([c for c in counts.index if all(counts.loc[c, y] >= min_days_required for y in years)])
    logger.info(f"Fixed panel identified: {len(panel)} cities with >= {min_days_required:.1f} days across all {years}.")
    return panel, wdf


def compute_bootstrap_ci(pct_values, n_boot=5000, seed=0):
    """Compute 95% bootstrap CI on median with seed 0."""
    rng = np.random.default_rng(seed)
    n = len(pct_values)
    boots = [float(np.median(rng.choice(pct_values, n))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def run_aqi_analysis(df, panel, wdf, years=ANALYSIS_YEARS):
    """Perform full statistical analysis across fixed panel cities."""
    panel_set = set(panel)
    panel_wdf = wdf[wdf['City'].isin(panel_set)].copy()

    # Per city and year: mean daily AQI in window
    city_means = panel_wdf.groupby(['City', 'Year'])['AQI Value'].mean().unstack(level=1)

    # Median across panel cities per year
    median_window_aqi = {}
    for y in years:
        median_window_aqi[str(y)] = round(float(city_means[y].median()), 2)

    # Pre-window context: 1 Jan to 24 Mar for panel cities
    m = df['Date'].dt.month
    d = df['Date'].dt.day
    pre_mask = (m == 1) | (m == 2) | ((m == 3) & (d <= 24))
    pre_df = df[pre_mask & df['City'].isin(panel_set)].copy()
    pre_df['Year'] = pre_df['Date'].dt.year
    pre_city_means = pre_df.groupby(['City', 'Year'])['AQI Value'].mean().unstack(level=1)

    pre_window_median_aqi = {}
    for y in years:
        if y in pre_city_means.columns:
            pre_window_median_aqi[str(y)] = round(float(pre_city_means[y].median()), 2)
        else:
            pre_window_median_aqi[str(y)] = None

    # Percent change vs 2020 per city, bootstrap CI, Wilcoxon test
    base_2020 = city_means[2020]
    pct_change_vs_2020 = {}

    for y in [2021, 2022, 2023, 2024, 2025]:
        series_y = city_means[y]
        pct = (series_y - base_2020) / base_2020 * 100.0
        med_pct = float(pct.median())
        lo, hi = compute_bootstrap_ci(pct.values, n_boot=5000, seed=0)
        w_res = stats.wilcoxon(series_y, base_2020, alternative='two-sided')

        pct_change_vs_2020[str(y)] = {
            'median_pct_change': round(med_pct, 2),
            'ci_low': round(lo, 2),
            'ci_high': round(hi, 2),
            'wilcoxon_stat': float(w_res.statistic),
            'wilcoxon_p': float(w_res.pvalue),
        }

    # Category shares on panel city-days in window:
    # Good+Satisfactory <=100, Moderate 101-200, Poor 201-300, Very Poor+Severe >300
    category_shares = {}
    for y in years:
        ydf = panel_wdf[panel_wdf['Year'] == y]['AQI Value']
        total = len(ydf)
        c_good_sat = int((ydf <= 100).sum())
        c_mod = int(((ydf > 100) & (ydf <= 200)).sum())
        c_poor = int(((ydf > 200) & (ydf <= 300)).sum())
        c_vp_sev = int((ydf > 300).sum())

        category_shares[str(y)] = {
            'total_city_days': total,
            'good_satisfactory_pct': round(c_good_sat / total * 100.0, 1),
            'moderate_pct': round(c_mod / total * 100.0, 1),
            'poor_pct': round(c_poor / total * 100.0, 1),
            'very_poor_severe_pct': round(c_vp_sev / total * 100.0, 1),
            'counts': {
                'good_satisfactory': c_good_sat,
                'moderate': c_mod,
                'poor': c_poor,
                'very_poor_severe': c_vp_sev,
            }
        }

    # Named cities window means
    named_cities_window_means = {}
    for c in NAMED_CITIES:
        if c in city_means.index:
            named_cities_window_means[c] = {
                str(y): round(float(city_means.loc[c, y]), 1) for y in years
            }
        else:
            logger.warning(f"Named city '{c}' not in panel.")

    return {
        'source': 'Kaggle krishnandansha/aqi-india-2020-to-2025-october (CPCB-derived, unverified)',
        'note': 'AQI only; 2025 data end 31 Oct 2025; fixed panel of 58 cities; 2022 window was the highest (cause not analysed); not comparable with the 2015-2020 results in /covid.',
        'database_table': 'daily_aqi (cache/daily_aqi.db)',
        'window_description': '25 Mar to 31 May (68 days)',
        'pre_window_description': '1 Jan to 24 Mar',
        'panel_size': len(panel),
        'panel_cities': panel,
        'years': years,
        'median_window_aqi': median_window_aqi,
        'pre_window_median_aqi': pre_window_median_aqi,
        'pct_change_vs_2020': pct_change_vs_2020,
        'category_shares': category_shares,
        'named_cities_window_means': named_cities_window_means,
    }


def sanity_check(results):
    """Check computed values against user's benchmarks and print warnings if off by >1 point."""
    logger.info("Executing sanity check against expected benchmarks...")
    panel_size = results['panel_size']
    if panel_size != SANITY_BENCHMARKS['panel_size']:
        logger.warning(f"SANITY WARNING: Panel size is {panel_size}, expected {SANITY_BENCHMARKS['panel_size']}")

    for y, exp_med in SANITY_BENCHMARKS['median_window_aqi'].items():
        act_med = results['median_window_aqi'][str(y)]
        diff = abs(act_med - exp_med)
        if diff > 1.0:
            logger.warning(f"SANITY WARNING: Median window AQI for {y} is {act_med}, expected {exp_med} (diff: {diff:.2f})")
        else:
            logger.info(f"Sanity OK: Year {y} median window AQI = {act_med:.1f} (expected {exp_med})")

    for y, exp_pct in SANITY_BENCHMARKS['median_pct_change'].items():
        act_pct = results['pct_change_vs_2020'][str(y)]['median_pct_change']
        diff = abs(act_pct - exp_pct)
        if diff > 1.0:
            logger.warning(f"SANITY WARNING: % change vs 2020 for {y} is {act_pct}%, expected {exp_pct}% (diff: {diff:.2f})")
        else:
            logger.info(f"Sanity OK: Year {y} % change vs 2020 = +{act_pct:.1f}% (expected +{exp_pct})")


def main():
    parser = argparse.ArgumentParser(description="Ingest and compute AQI trends for 2020-2025.")
    parser.add_argument('--input', '-i', help="Path to AQI_2020_to_2025_October.csv")
    parser.add_argument('--output', '-o', default=OUTPUT_JSON, help="Path for aqi_trends.json")
    parser.add_argument('--db', default=DB_PATH, help="Path for SQLite database")
    args = parser.parse_args()

    csv_path = find_csv_path(args.input)
    df = load_and_clean_csv(csv_path)

    store_in_sqlite(df, args.db)

    panel, wdf = compute_fixed_panel(df)
    results = run_aqi_analysis(df, panel, wdf)

    sanity_check(results)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Successfully wrote AQI trends to {args.output}")
    print(f"Panel size: {results['panel_size']}")
    print(f"Panel cities: {results['panel_cities']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
