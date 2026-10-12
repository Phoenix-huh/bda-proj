"""All routes for the four novelties. Registered from app.py via a Blueprint."""
import base64
import datetime
import json
import logging
import os
import sqlite3
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
from flask import Blueprint, abort, jsonify, render_template, request, send_from_directory
from matplotlib.figure import Figure

import covid
from advisory import advisory_for, aqi_bucket
from features import ALL_POLLUTANTS, build_station_frame

bp = Blueprint('novelties', __name__)
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))


def p(*parts):
    return os.path.join(BASE, *parts)


DB_PATH = p('cache', 'hourly.db')
RESULTS = p('results')
AQI_BANDS = [(0, 50, '#00b050'), (50, 100, '#92d050'), (100, 200, '#ffd400'),
             (200, 300, '#ff9900'), (300, 400, '#ff0000'), (400, 500, '#990000')]

_fc = {}
_city = {}
_covid = {}


# ------------------------------------------------------------------ helpers
def records(df):
    """DataFrame -> list of dicts with NaN converted to null."""
    return json.loads(df.to_json(orient='records'))


def read_csv_records(name):
    path = p('results', name)
    return records(pd.read_csv(path)) if os.path.exists(path) else []


def get_forecast():
    if _fc.get('ready'):
        return _fc
    try:
        bundle = joblib.load(p('models', 'forecast_models.joblib'))
        with open(p('models', 'forecast_meta.json')) as f:
            meta = json.load(f)
        _fc.update(models=bundle['models'], conformal=bundle['conformal'], meta=meta, ready=True)
    except Exception as e:  # models not trained yet
        _fc.clear()
        _fc.update(ready=False, error=str(e))
    return _fc


def station_options(only=None):
    st = pd.read_csv(p('data', 'stations.csv'))[['StationId', 'City', 'State']].dropna()
    if only is not None:
        st = st[st['StationId'].isin(only)]
    return [{'StationId': r.StationId, 'display': f'{r.City}, {r.State} ({r.StationId})'}
            for r in st.itertuples()]


def fmt(ts):
    return ts.strftime('%Y-%m-%d %H:00:00')


def fetch_rows(station, start, end):
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError('cache/hourly.db is missing. Run build_cache.py first.')
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            'SELECT * FROM hourly WHERE StationId=? AND Datetime BETWEEN ? AND ?',
            con, params=(station, fmt(start), fmt(end)))
    finally:
        con.close()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df


def feature_label(col):
    simple = {'Hour': 'Hour of day', 'Month': 'Month', 'DayOfWeek': 'Day of week',
              'doy_sin': 'Season (day of year)', 'doy_cos': 'Season (day of year, cos)',
              'StationId': 'Station'}
    if col in simple:
        return simple[col]
    kind, name = col.split('_', 1)
    return {'cur': f'{name} now', 'lag6': f'{name} 6 h earlier',
            'lag24': f'{name} 24 h earlier', 'roll24': f'{name} 24 h average'}[kind]


# ------------------------------------------------------------------ forecast core
def forecast_for(station, origin, target, explain_h):
    fc = get_forecast()
    meta, models, conf = fc['meta'], fc['models'], fc['conformal']
    rows = fetch_rows(station, origin - pd.Timedelta(hours=48), origin + pd.Timedelta(hours=48))

    win = pd.DataFrame({'Datetime': pd.date_range(origin - pd.Timedelta(hours=30), origin, freq='h')})
    win = win.merge(rows, on='Datetime', how='left')
    feats = build_station_frame(win, horizons=[], targets=[]).iloc[-1]
    if feats[[f'cur_{c}' for c in ALL_POLLUTANTS]].isna().all():
        raise LookupError('No observation for this station at that hour. Try another time or use "Use latest data".')

    X = pd.DataFrame({c: [float(feats[c])] for c in meta['feature_cols'] if c != 'StationId'})
    X.insert(0, 'StationId', pd.Categorical([station], categories=meta['station_ids']))
    X = X[meta['feature_cols']]

    forecasts = {}
    for t in meta['targets']:
        forecasts[t] = []
        for h in meta['horizons']:
            q = sorted(float(models[f'{t}|{h}|{qq}'].predict(X)[0]) for qq in meta['quantiles'])
            qhat = conf[f'{t}|{h}']
            med = max(q[1], 0.0)
            lo = min(max(0.0, q[0] - qhat), med)
            hi = max(q[2] + qhat, med)
            when = origin + pd.Timedelta(hours=h)
            hit = rows.loc[rows['Datetime'] == when, t]
            actual = float(hit.iloc[0]) if len(hit) and pd.notna(hit.iloc[0]) else None
            item = {'horizon': h, 'time': when.strftime('%Y-%m-%d %H:%M'), 'lower': lo,
                    'median': med, 'upper': hi, 'actual': actual}
            if t == 'AQI':
                item.update(bucket_lower=aqi_bucket(lo), bucket_median=aqi_bucket(med),
                            bucket_upper=aqi_bucket(hi))
            forecasts[t].append(item)

    # SHAP-style explanation of the median model (LightGBM exact TreeSHAP)
    key = f'{target}|{explain_h}|0.5'
    contrib = models[key].predict(X, pred_contrib=True)[0]
    names = meta['feature_cols']
    items = []
    for i, c in enumerate(names):
        v = X[c].iloc[0]
        items.append({'feature': c, 'label': feature_label(c),
                      'value': str(v) if c == 'StationId' else round(float(v), 2) if pd.notna(v) else None,
                      'shap': float(contrib[i])})
    items.sort(key=lambda d: -abs(d['shap']))
    explanation = {'target': target, 'horizon': explain_h, 'base_value': float(contrib[-1]),
                   'top': items[:8]}

    advisory = None
    if 'AQI' in forecasts:
        a = next((x for x in forecasts['AQI'] if x['horizon'] == explain_h), forecasts['AQI'][0])
        pm = next((x for x in forecasts.get('PM2.5', []) if x['horizon'] == a['horizon']), None)
        advisory = advisory_for(a['median'], pm['median'] if pm else None)
        advisory['for_horizon'] = a['horizon']

    return rows, forecasts, explanation, advisory


def forecast_chart(rows, origin, target, items):
    hist = rows[(rows['Datetime'] <= origin) & (rows['Datetime'] >= origin - pd.Timedelta(hours=48))]
    cur0 = hist.loc[hist['Datetime'] == origin, target]
    cur0 = float(cur0.iloc[0]) if len(cur0) and pd.notna(cur0.iloc[0]) else None

    times = [origin + pd.Timedelta(hours=i['horizon']) for i in items]
    lo = [i['lower'] for i in items]
    med = [i['median'] for i in items]
    hi = [i['upper'] for i in items]
    if cur0 is not None:
        times, lo, med, hi = [origin] + times, [cur0] + lo, [cur0] + med, [cur0] + hi

    fig = Figure(figsize=(10, 4.8))
    ax = fig.subplots()
    if target == 'AQI':
        for a, b, c in AQI_BANDS:
            ax.axhspan(a, b, color=c, alpha=0.07)
    ax.plot(hist['Datetime'], hist[target], color='#334e68', label='Observed (past 48 h)')
    ax.fill_between(times, lo, hi, color='#1f6feb', alpha=0.25, label='80% prediction interval')
    ax.plot(times, med, '-o', color='#1f6feb', label='Forecast (median)')
    act = [(origin + pd.Timedelta(hours=i['horizon']), i['actual']) for i in items if i['actual'] is not None]
    if act:
        ax.scatter([a[0] for a in act], [a[1] for a in act], color='#d62828', zorder=5, label='Actual (observed)')
    ax.axvline(origin, color='gray', linestyle='--', linewidth=1)
    ax.set_ylim(bottom=0)
    ax.set_title(f'{target} forecast from {origin:%Y-%m-%d %H:00}')
    ax.set_ylabel(target)
    ax.legend(loc='upper left', fontsize=8)
    fig.autofmt_xdate()
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode('ascii')


# ------------------------------------------------------------------ forecast routes
@bp.route('/forecast')
def forecast_page():
    fc = get_forecast()
    if fc.get('ready'):
        meta = fc['meta']
        return render_template('forecast.html', ready=True, error=None,
                               stations=station_options(meta['station_ids']),
                               targets=meta['targets'], horizons=meta['horizons'],
                               level=int(meta['interval_level'] * 100))
    return render_template('forecast.html', ready=False, error=fc.get('error'),
                           stations=[], targets=[], horizons=[], level=80)


@bp.route('/api/station-range')
def station_range():
    station = request.args.get('station', '')
    if not os.path.exists(DB_PATH):
        return jsonify({'error': 'cache/hourly.db missing'}), 503
    con = sqlite3.connect(DB_PATH)
    try:
        lo, hi = con.execute('SELECT MIN(Datetime), MAX(Datetime) FROM hourly WHERE StationId=?',
                             (station,)).fetchone()
    finally:
        con.close()
    if not lo:
        return jsonify({'error': 'No data for this station'}), 404
    return jsonify({'min': lo, 'max': hi})


@bp.route('/api/forecast', methods=['POST'])
def api_forecast():
    fc = get_forecast()
    if not fc.get('ready'):
        return jsonify({'error': 'Forecast models not trained yet. Run: python train_forecast.py'}), 503
    meta = fc['meta']
    station = request.form.get('station', '').strip()
    target = request.form.get('target', 'AQI')
    try:
        explain_h = int(request.form.get('explain_horizon', 24))
        hour = int(request.form.get('hour', ''))
        if not 0 <= hour <= 23:
            raise ValueError
        origin = datetime.datetime.strptime(request.form.get('date', ''), '%Y-%m-%d').replace(hour=hour)
    except ValueError:
        return jsonify({'error': 'Provide a valid date (YYYY-MM-DD) and hour (0-23).'}), 400
    origin = pd.Timestamp(origin)
    if station not in meta['station_ids']:
        return jsonify({'error': 'Unknown station'}), 400
    if target not in meta['targets'] or explain_h not in meta['horizons']:
        return jsonify({'error': 'Unsupported target or horizon'}), 400

    try:
        rows, forecasts, explanation, advisory = forecast_for(station, origin, target, explain_h)
        return jsonify({
            'origin': origin.strftime('%Y-%m-%d %H:%M'),
            'interval_level': meta['interval_level'],
            'forecasts': forecasts,
            'chart': forecast_chart(rows, origin, target, forecasts[target]),
            'explanation': explanation,
            'advisory': advisory,
        })
    except LookupError as e:
        return jsonify({'error': str(e)}), 404
    except Exception:
        logger.exception('Forecast failed')
        return jsonify({'error': 'Forecast failed. See server log.'}), 500


# ------------------------------------------------------------------ models page (Novelty 1)
@bp.route('/models')
def models_page():
    shap_images = sorted(f for f in os.listdir(RESULTS) if f.startswith('shap_') and f.endswith('.png')) \
        if os.path.isdir(RESULTS) else []
    return render_template('models.html',
                           comparison=read_csv_records('model_comparison.csv'),
                           coverage=read_csv_records('interval_coverage.csv'),
                           shap_images=shap_images)


@bp.route('/results/<path:name>')
def results_file(name):
    if not name.lower().endswith(('.png', '.csv', '.json')):
        abort(404)
    return send_from_directory(RESULTS, name)


# ------------------------------------------------------------------ COVID page (Novelty 3)
def get_covid_df():
    if 'df' not in _covid:
        _covid['df'] = covid.load_city_day(p('data', 'city_day.csv'))
    return _covid['df']


@bp.route('/covid')
def covid_page():
    df = get_covid_df()
    cities = ['All cities'] + sorted(df['City'].astype(str).unique())
    buckets_path = p('results', 'covid_buckets.json')
    buckets = json.load(open(buckets_path)) if os.path.exists(buckets_path) else []
    return render_template('covid.html', cities=cities, pollutants=covid.POLLUTANTS,
                           summary=read_csv_records('covid_summary.csv'),
                           counterfactual=read_csv_records('covid_counterfactual_summary.csv'),
                           buckets=buckets)


@bp.route('/api/covid-chart')
def covid_chart():
    city = request.args.get('city', 'All cities')
    pollutant = request.args.get('pollutant', 'PM2.5')
    df = get_covid_df()
    if pollutant not in covid.POLLUTANTS or (city != 'All cities' and city not in set(df['City'].astype(str))):
        return jsonify({'error': 'Unknown city or pollutant'}), 400
    png = covid.render_chart_png(df, city, pollutant)
    return jsonify({'chart': base64.b64encode(png).decode('ascii')})


# ------------------------------------------------------------------ map (Novelty 4)
def get_city_daily():
    if 'df' not in _city:
        daily = pd.read_csv(p('cache', 'station_daily.csv'), parse_dates=['Date'])
        st = pd.read_csv(p('data', 'stations.csv'))[['StationId', 'City']].dropna()
        coords = pd.read_csv(p('data', 'city_coords.csv'))
        m = daily.merge(st, on='StationId', how='inner')
        city = m.groupby(['City', 'Date'])[ALL_POLLUTANTS].mean().reset_index()
        missing = sorted(set(city['City']) - set(coords['City']))
        if missing:
            logger.warning('No coordinates for cities (skipped on map): %s', missing)
        _city['df'] = city.merge(coords, on='City', how='inner')
    return _city['df']


@bp.route('/map')
def map_page():
    df = get_city_daily()
    return render_template('map.html', min_date=str(df['Date'].min().date()),
                           max_date=str(df['Date'].max().date()), pollutants=covid.POLLUTANTS)


@bp.route('/api/map-data')
def map_data():
    layer = request.args.get('layer', 'aqi')
    df = get_city_daily()

    if layer == 'covid':
        pollutant = request.args.get('pollutant', 'PM2.5')
        period = request.args.get('period', 'lockdown')
        path = p('results', 'covid_yoy.csv')
        if not os.path.exists(path):
            return jsonify({'error': 'Run covid.py first'}), 503
        yoy = pd.read_csv(path)
        yoy = yoy[(yoy['Pollutant'] == pollutant) & (yoy['Period'] == period)]
        coords = df[['City', 'Lat', 'Lon']].drop_duplicates('City')
        m = yoy.merge(coords, on='City', how='inner')
        pts = [{'city': r.City, 'lat': r.Lat, 'lon': r.Lon, 'pct': round(float(r.PctChange), 1),
                'mean2020': round(float(r.Mean2020), 1), 'meanBase': round(float(r.MeanBase), 1)}
               for r in m.itertuples()]
        return jsonify({'layer': 'covid', 'pollutant': pollutant, 'period': period, 'points': pts})

    try:
        date = pd.Timestamp(datetime.datetime.strptime(request.args.get('date', ''), '%Y-%m-%d'))
    except ValueError:
        return jsonify({'error': 'date must be YYYY-MM-DD'}), 400
    day = df[(df['Date'] == date) & df['AQI'].notna()]
    pts = []
    for r in day.itertuples():
        row = day.loc[r.Index]
        pm = float(row['PM2.5']) if pd.notna(row['PM2.5']) else None
        adv = advisory_for(float(row['AQI']), pm)
        pts.append({'city': r.City, 'lat': r.Lat, 'lon': r.Lon, 'aqi': round(float(row['AQI'])),
                    'bucket': adv['bucket'], 'color': adv['color'], 'advisory': adv,
                    'pollutants': {k: (round(float(row[k]), 1) if pd.notna(row[k]) else None)
                                   for k in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']}})
    return jsonify({'layer': 'aqi', 'date': date.strftime('%Y-%m-%d'), 'points': pts})


@bp.route('/api/advisory')
def api_advisory():
    try:
        aqi = float(request.args.get('aqi', ''))
    except ValueError:
        return jsonify({'error': 'aqi must be a number'}), 400
    pm25 = request.args.get('pm25')
    return jsonify(advisory_for(aqi, float(pm25) if pm25 else None))
