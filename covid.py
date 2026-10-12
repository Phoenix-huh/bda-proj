"""
Novelty 3: COVID-19 lockdown impact analysis on city-level daily data.

    python covid.py        # writes results/covid_*.csv / .json
"""
import json
import os
from io import BytesIO

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

RESULTS = 'results'
POLLUTANTS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
              'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
BASE_YEARS = [2017, 2018, 2019]
PERIODS = {'lockdown': ('03-25', '05-31'), 'unlock1': ('06-01', '06-30')}
CUTOFF_MD = '03-25'
MIN_DAYS = 15


def load_city_day(path=os.path.join('data', 'city_day.csv')):
    df = pd.read_csv(path, parse_dates=['Date'])
    df['City'] = df['City'].astype('category')
    return df


def _window(df, year, period):
    s, e = PERIODS[period]
    return df[(df['Date'] >= pd.Timestamp(f'{year}-{s}')) & (df['Date'] <= pd.Timestamp(f'{year}-{e}'))]


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (NaNs are passed through)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


# ---------------------------------------------------------------- A. year-on-year
def yoy_table(df, base_years=BASE_YEARS):
    rows = []
    for period in PERIODS:
        w20 = _window(df, 2020, period)
        wb = pd.concat([_window(df, y, period) for y in base_years])
        for p in POLLUTANTS:
            a = w20.groupby('City', observed=True)[p].agg(['mean', 'count'])
            b = wb.groupby('City', observed=True)[p].agg(['mean', 'count'])
            j = a.join(b, lsuffix='_2020', rsuffix='_base', how='inner')
            j = j[(j['count_2020'] >= MIN_DAYS) & (j['count_base'] >= MIN_DAYS) & (j['mean_base'] > 0)]
            for city, r in j.iterrows():
                rows.append({'City': city, 'Pollutant': p, 'Period': period,
                             'Mean2020': r['mean_2020'], 'MeanBase': r['mean_base'],
                             'PctChange': (r['mean_2020'] - r['mean_base']) / r['mean_base'] * 100,
                             'Days2020': int(r['count_2020']), 'DaysBase': int(r['count_base'])})
    return pd.DataFrame(rows)


def summarize(yoy, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for (period, p), g in yoy.groupby(['Period', 'Pollutant']):
        if len(g) < 5:
            continue
        pct = g['PctChange'].values
        boots = [np.median(rng.choice(pct, len(pct))) for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        try:
            pval = float(stats.wilcoxon(g['Mean2020'], g['MeanBase']).pvalue)
        except ValueError:
            pval = np.nan
        rows.append({'Period': period, 'Pollutant': p, 'n_cities': len(g),
                     'MedianPctChange': float(np.median(pct)), 'CI_low': lo, 'CI_high': hi,
                     'p_value': pval})
    out = pd.DataFrame(rows)
    out['p_adj_BH'] = np.nan
    for period in out['Period'].unique():
        m = out['Period'] == period
        out.loc[m, 'p_adj_BH'] = bh_adjust(out.loc[m, 'p_value'].values)
    return out


def bucket_shift(df, base_years=BASE_YEARS):
    good, bad = ['Good', 'Satisfactory'], ['Poor', 'Very Poor', 'Severe']
    out = []
    for period in PERIODS:
        w20 = _window(df, 2020, period).dropna(subset=['AQI_Bucket'])
        wb = pd.concat([_window(df, y, period) for y in base_years]).dropna(subset=['AQI_Bucket'])
        if len(w20) == 0 or len(wb) == 0:
            continue
        out.append({'period': period,
                    'good_or_satisfactory_2020_pct': round(float(w20['AQI_Bucket'].isin(good).mean() * 100), 1),
                    'good_or_satisfactory_baseline_pct': round(float(wb['AQI_Bucket'].isin(good).mean() * 100), 1),
                    'poor_or_worse_2020_pct': round(float(w20['AQI_Bucket'].isin(bad).mean() * 100), 1),
                    'poor_or_worse_baseline_pct': round(float(wb['AQI_Bucket'].isin(bad).mean() * 100), 1),
                    'n_city_days_2020': len(w20), 'n_city_days_baseline': len(wb)})
    return out


# ---------------------------------------------------------------- B. ML counterfactual
def counterfactual(df, test_year):
    from lightgbm import LGBMRegressor
    cutoff = pd.Timestamp(f'{test_year}-{CUTOFF_MD}')
    d = df.copy()
    doy = d['Date'].dt.dayofyear
    d['doy_sin'] = np.sin(2 * np.pi * doy / 366)
    d['doy_cos'] = np.cos(2 * np.pi * doy / 366)
    d['dow'] = d['Date'].dt.dayofweek
    d['t'] = (d['Date'] - pd.Timestamp('2015-01-01')).dt.days / 365.25
    feat = ['City', 'doy_sin', 'doy_cos', 'dow', 't']

    rows = []
    for p in POLLUTANTS:
        sub = d.dropna(subset=[p])
        tr = sub[sub['Date'] < cutoff]
        if len(tr) < 500:
            continue
        m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                          min_child_samples=20, verbose=-1, random_state=0)
        m.fit(tr[feat], tr[p])
        for period, (s, e) in PERIODS.items():
            te = sub[(sub['Date'] >= pd.Timestamp(f'{test_year}-{s}')) &
                     (sub['Date'] <= pd.Timestamp(f'{test_year}-{e}'))]
            if te.empty:
                continue
            te = te.assign(Expected=m.predict(te[feat]))
            g = te.groupby('City', observed=True).agg(Actual=(p, 'mean'), Expected=('Expected', 'mean'),
                                                      n=(p, 'count'))
            g = g[(g['n'] >= MIN_DAYS) & (g['Expected'] > 0)]
            for city, r in g.iterrows():
                rows.append({'Year': test_year, 'City': city, 'Pollutant': p, 'Period': period,
                             'Actual': r['Actual'], 'Expected': r['Expected'],
                             'PctDev': (r['Actual'] - r['Expected']) / r['Expected'] * 100})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- charts
def chart_series(df, city, pollutant, base_years=BASE_YEARS):
    d = df if city == 'All cities' else df[df['City'] == city]
    d = d.dropna(subset=[pollutant])
    daily = d.groupby('Date', observed=True)[pollutant].mean()
    full = daily.reindex(pd.date_range('2015-01-01', '2020-07-01'))
    sm = full.rolling(7, min_periods=3, center=True).mean()
    sm = sm[~((sm.index.month == 2) & (sm.index.day == 29))]
    tbl = (pd.DataFrame({'Year': sm.index.year, 'MD': sm.index.strftime('%m-%d'), 'v': sm.values})
           .pivot(index='MD', columns='Year', values='v'))
    tbl = tbl[tbl.index <= '08-31']
    base = tbl[[y for y in base_years if y in tbl.columns]]
    y20 = tbl[2020] if 2020 in tbl.columns else pd.Series(np.nan, index=tbl.index)
    x = pd.to_datetime(['2020-' + md for md in tbl.index])
    return x, y20.values, base.mean(axis=1).values, base.min(axis=1).values, base.max(axis=1).values


def render_chart_png(df, city, pollutant):
    x, y20, bmean, bmin, bmax = chart_series(df, city, pollutant)
    fig = Figure(figsize=(10, 4.8))
    ax = fig.subplots()
    ax.fill_between(x, bmin, bmax, color='#9fb3c8', alpha=0.4,
                    label=f'{BASE_YEARS[0]}-{BASE_YEARS[-1]} range')
    ax.plot(x, bmean, color='#486581', label=f'{BASE_YEARS[0]}-{BASE_YEARS[-1]} mean')
    ax.plot(x, y20, color='#d62828', linewidth=2, label='2020')
    ax.axvspan(pd.Timestamp('2020-03-25'), pd.Timestamp('2020-05-31'), color='#f0b429', alpha=0.2,
               label='Lockdown (25 Mar - 31 May)')
    ax.set_title(f'{pollutant}: {city} (7-day rolling mean)')
    ax.set_ylabel(pollutant)
    ax.legend(loc='upper right', fontsize=8)
    fig.autofmt_xdate()
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    return buf.getvalue()


# ---------------------------------------------------------------- driver
def run_all():
    os.makedirs(RESULTS, exist_ok=True)
    df = load_city_day()
    print(f"city_day: {len(df):,} rows, {df['City'].nunique()} cities, "
          f"{df['Date'].min().date()} -> {df['Date'].max().date()}")

    yoy = yoy_table(df)
    yoy.round(3).to_csv(os.path.join(RESULTS, 'covid_yoy.csv'), index=False)
    summ = summarize(yoy)
    summ.round(4).to_csv(os.path.join(RESULTS, 'covid_summary.csv'), index=False)

    cf = pd.concat([counterfactual(df, 2020), counterfactual(df, 2019)], ignore_index=True)
    cf.round(3).to_csv(os.path.join(RESULTS, 'covid_counterfactual.csv'), index=False)
    cf_sum = (cf.groupby(['Year', 'Period', 'Pollutant'])
                .agg(MedianPctDev=('PctDev', 'median'), n_cities=('City', 'nunique')).reset_index())
    cf_sum.round(3).to_csv(os.path.join(RESULTS, 'covid_counterfactual_summary.csv'), index=False)

    with open(os.path.join(RESULTS, 'covid_buckets.json'), 'w') as f:
        json.dump(bucket_shift(df), f, indent=2)

    print("\nMedian % change vs 2017-2019 (lockdown window):")
    print(summ[summ.Period == 'lockdown'][['Pollutant', 'n_cities', 'MedianPctChange', 'CI_low', 'CI_high', 'p_adj_BH']]
          .round(2).to_string(index=False))
    print("\nCounterfactual median % deviation, lockdown window (2020 vs 2019 placebo):")
    piv = cf_sum[cf_sum.Period == 'lockdown'].pivot(index='Pollutant', columns='Year', values='MedianPctDev').round(1)
    print(piv.to_string())


if __name__ == '__main__':
    run_all()
