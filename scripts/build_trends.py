"""Build own trends analysis for 2021 onward from CPCB CSVs.

Reads new CPCB data placed in data/new/, cleans it, computes city-level
lockdown-window (25 Mar - 31 May) averages for each year 2017-latest,
and calculates:
  - Median across cities
  - Percent change vs 2020 and vs 2017-2019 baseline
  - Bootstrap 95% CI (5000 resamples)
  - Wilcoxon signed-rank test
  - Benjamini-Hochberg adjusted p-values
Writes output to data/own_trends.json.
"""
import argparse
import datetime
import glob
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from covid import BASE_YEARS, MIN_DAYS, bh_adjust

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurable column mapping
DEFAULT_COLUMN_MAPPING = {
    'station': 'StationId',
    'stationid': 'StationId',
    'station_id': 'StationId',
    'station name': 'StationName',
    'city': 'City',
    'state': 'State',
    'datetime': 'Datetime',
    'date': 'Date',
    'pm2.5': 'PM2.5',
    'pm25': 'PM2.5',
    'pm10': 'PM10',
    'no2': 'NO2',
    'no': 'NO',
    'nox': 'NOx',
    'nh3': 'NH3',
    'co': 'CO',
    'so2': 'SO2',
    'o3': 'O3',
    'ozone': 'O3',
    'aqi': 'AQI',
}

POLLUTANTS_OF_INTEREST = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
IMPLAUSIBLE_ZERO_POLLUTANTS = {'PM2.5', 'PM10', 'NO2', 'O3'}


def clean_dataframe(df, col_map=None):
    """Normalize columns, clean placeholders, negative values, and duplicates."""
    mapping = dict(DEFAULT_COLUMN_MAPPING)
    if col_map:
        mapping.update(col_map)

    # Standardize column names
    rename_dict = {}
    for col in df.columns:
        norm = str(col).strip().lower()
        if norm in mapping:
            rename_dict[col] = mapping[norm]
        else:
            rename_dict[col] = col.strip()
    df = df.rename(columns=rename_dict)

    if 'City' not in df.columns:
        raise ValueError("Dataset must contain a 'City' column.")

    # Parse date / datetime
    if 'Date' not in df.columns and 'Datetime' in df.columns:
        df['Date'] = pd.to_datetime(df['Datetime'], errors='coerce').dt.date
        df['Date'] = pd.to_datetime(df['Date'])
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        raise ValueError("Dataset must contain 'Date' or 'Datetime' column.")

    df = df.dropna(subset=['Date', 'City'])

    # Drop duplicate records
    dup_cols = [c for c in ['StationId', 'Datetime', 'Date', 'City'] if c in df.columns]
    if len(dup_cols) >= 2:
        df = df.drop_duplicates(subset=dup_cols, keep='last')

    # Clean numeric pollutant columns
    present_pollutants = [p for p in POLLUTANTS_OF_INTEREST if p in df.columns]
    for p in present_pollutants:
        df[p] = pd.to_numeric(df[p], errors='coerce')
        # Drop negatives
        df.loc[df[p] < 0, p] = np.nan
        # Drop 999 placeholders (>= 990)
        df.loc[df[p] >= 990, p] = np.nan
        # Drop 0s where implausible
        if p in IMPLAUSIBLE_ZERO_POLLUTANTS:
            df.loc[df[p] == 0, p] = np.nan

    return df, present_pollutants


def filter_lockdown_window(df):
    """Filter rows to the lockdown period: 25 Mar - 31 May for each year."""
    m = df['Date'].dt.month
    d = df['Date'].dt.day
    mask = ((m == 3) & (d >= 25)) | (m == 4) | (m == 5)
    w = df[mask].copy()
    w['Year'] = w['Date'].dt.year
    return w


def compute_bootstrap_ci(pct_values, n_boot=5000, seed=42):
    """Compute 95% bootstrap confidence interval on the median."""
    if len(pct_values) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = [float(np.median(rng.choice(pct_values, len(pct_values)))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def run_trend_analysis(df, present_pollutants, min_days=MIN_DAYS):
    """Compute city-level lockdown window means, medians, paired % changes and statistics."""
    window_df = filter_lockdown_window(df)
    years = sorted(window_df['Year'].unique())
    logger.info(f"Years found in lockdown window: {years}")

    # Compute city-level means per year and pollutant with day count check
    city_year_means = {}
    city_year_counts = {}
    for (city, year), grp in window_df.groupby(['City', 'Year']):
        for p in present_pollutants:
            series = grp[p].dropna()
            count = len(series)
            if count >= min_days:
                city_year_means.setdefault(p, {})[(city, year)] = float(series.mean())
                city_year_counts.setdefault(p, {})[(city, year)] = count

    # Baseline 2017-2019 mean per city
    city_base_means = {}
    base_window = window_df[window_df['Year'].isin(BASE_YEARS)]
    for city, grp in base_window.groupby('City'):
        for p in present_pollutants:
            series = grp[p].dropna()
            if len(series) >= min_days:
                city_base_means.setdefault(p, {})[city] = float(series.mean())

    results = {
        'years': [int(y) for y in years if y >= 2021],
        'all_years': [int(y) for y in years],
        'pollutants': present_pollutants,
        'city_medians': {},
        'vs_2020': {},
        'vs_baseline': {},
    }

    # Absolute city medians per year
    for p in present_pollutants:
        results['city_medians'][p] = {}
        for y in years:
            vals = [val for (c, yr), val in city_year_means.get(p, {}).items() if yr == y]
            if vals:
                results['city_medians'][p][str(y)] = round(float(np.median(vals)), 2)

    # Comparisons for target years (2021 onward) vs 2020 and vs Baseline
    target_years = [y for y in years if y >= 2021]

    for p in present_pollutants:
        vs_2020_list = []
        vs_base_list = []
        p_means = city_year_means.get(p, {})
        base_means = city_base_means.get(p, {})

        # 2020 city lookup
        cities_2020 = {c: val for (c, yr), val in p_means.items() if yr == 2020 and val > 0}

        for y in target_years:
            cities_y = {c: val for (c, yr), val in p_means.items() if yr == y and val > 0}

            # vs 2020
            common_2020 = sorted(set(cities_y.keys()) & set(cities_2020.keys()))
            if len(common_2020) >= 3:
                y_vals = np.array([cities_y[c] for c in common_2020])
                c2020_vals = np.array([cities_2020[c] for c in common_2020])
                pct_changes = (y_vals - c2020_vals) / c2020_vals * 100.0
                med_pct = float(np.median(pct_changes))
                lo, hi = compute_bootstrap_ci(pct_changes)
                try:
                    pval = float(stats.wilcoxon(y_vals, c2020_vals).pvalue)
                except Exception:
                    pval = np.nan
                vs_2020_list.append({
                    'year': int(y),
                    'n_cities': len(common_2020),
                    'median_pct_change': round(med_pct, 2),
                    'ci_low': round(lo, 2) if not np.isnan(lo) else None,
                    'ci_high': round(hi, 2) if not np.isnan(hi) else None,
                    'p_value': round(pval, 5) if not np.isnan(pval) else None,
                })

            # vs Baseline (2017-2019)
            common_base = sorted(set(cities_y.keys()) & set(base_means.keys()))
            if len(common_base) >= 3:
                y_vals = np.array([cities_y[c] for c in common_base])
                base_vals = np.array([base_means[c] for c in common_base])
                pct_changes = (y_vals - base_vals) / base_vals * 100.0
                med_pct = float(np.median(pct_changes))
                lo, hi = compute_bootstrap_ci(pct_changes)
                try:
                    pval = float(stats.wilcoxon(y_vals, base_vals).pvalue)
                except Exception:
                    pval = np.nan
                vs_base_list.append({
                    'year': int(y),
                    'n_cities': len(common_base),
                    'median_pct_change': round(med_pct, 2),
                    'ci_low': round(lo, 2) if not np.isnan(lo) else None,
                    'ci_high': round(hi, 2) if not np.isnan(hi) else None,
                    'p_value': round(pval, 5) if not np.isnan(pval) else None,
                })

        # Apply Benjamini-Hochberg adjustment
        for lst in [vs_2020_list, vs_base_list]:
            pvals = [item['p_value'] if item['p_value'] is not None else np.nan for item in lst]
            if pvals:
                adj = bh_adjust(pvals)
                for item, a in zip(lst, adj):
                    item['p_adj_BH'] = round(float(a), 5) if not np.isnan(a) else None

        results['vs_2020'][p] = vs_2020_list
        results['vs_baseline'][p] = vs_base_list

    return results


def main():
    parser = argparse.ArgumentParser(description="Process CPCB CSV to build trends for 2021 onward.")
    parser.add_argument('--input', '-i', help="Path to input CSV. Defaults to scanning data/new/*.csv.")
    parser.add_argument('--output', '-o', default=os.path.join(BASE_DIR, 'data', 'own_trends.json'),
                        help="Path to write own_trends.json")
    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        new_dir = os.path.join(BASE_DIR, 'data', 'new')
        csv_files = glob.glob(os.path.join(new_dir, '*.csv'))
        if not csv_files:
            logger.warning(f"No CSV found in {new_dir}. Place CPCB CSV files in data/new/ to run.")
            return 0
        input_path = csv_files[0]
        logger.info(f"Using input CSV: {input_path}")

    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return 1

    logger.info(f"Loading data from {input_path}...")
    raw_df = pd.read_csv(input_path, low_memory=False)
    cleaned_df, pollutants = clean_dataframe(raw_df)
    logger.info(f"Cleaned {len(cleaned_df)} rows. Pollutants found: {pollutants}")

    results = run_trend_analysis(cleaned_df, pollutants)
    results['generated_at'] = datetime.datetime.now().isoformat()
    results['source_file'] = os.path.basename(input_path)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Successfully generated {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

