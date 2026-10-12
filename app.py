import base64
import datetime
import json
import logging
import os
import sqlite3
from io import BytesIO

import joblib
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))


def path(*parts):
    return os.path.join(BASE, *parts)


MODEL_PATH = os.environ.get('MODEL_PATH', path('models', 'air_quality_model.pkl'))
DB_PATH = path('cache', 'hourly.db')

TARGETS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
           'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(path('app.log')), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ----------------------------------------------------------------------
# Startup: load model + small cache files only (no 222 MB CSV)
# ----------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
logger.info(f"Model loaded from {MODEL_PATH}")

stations = pd.read_csv(path('data', 'stations.csv'))
station_list = [
    {'StationId': r.StationId, 'display': f"{r.City}, {r.State} ({r.StationId})"}
    for r in stations[['StationId', 'City', 'State']].dropna().itertuples()
]
valid_stations = {s['StationId'] for s in station_list}

with open(path('cache', 'global_means.json')) as f:
    global_means = json.load(f)

station_means = pd.read_csv(path('cache', 'station_means.csv')).set_index('StationId').to_dict('index')
station_hour_means = (pd.read_csv(path('cache', 'station_hour_means.csv'))
                      .set_index(['StationId', 'Hour']).to_dict('index'))
_daily = pd.read_csv(path('cache', 'station_daily.csv'), parse_dates=['Date'])
daily_by_station = {sid: g.set_index('Date').sort_index() for sid, g in _daily.groupby('StationId')}

clip_bounds = {}
if os.path.exists(path('models', 'clip_bounds.json')):
    with open(path('models', 'clip_bounds.json')) as f:
        clip_bounds = json.load(f)
else:
    logger.warning("models/clip_bounds.json missing; retrain with train_model.py")

logger.info(f"Ready: {len(valid_stations)} stations")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def error(msg, code=400):
    return jsonify({'error': msg}), code


def aqi_bucket(aqi):
    if aqi <= 50:
        return 'Good'
    if aqi <= 100:
        return 'Satisfactory'
    if aqi <= 200:
        return 'Moderate'
    if aqi <= 300:
        return 'Poor'
    if aqi <= 400:
        return 'Very Poor'
    return 'Severe'


def observed_previous_hour(station, when):
    """Actual reading at (station, when - 1h) from SQLite, or {}."""
    if not os.path.exists(DB_PATH):
        return {}
    prev = (when - datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:00:00')
    cols = ','.join(f'"{t}"' for t in TARGETS)
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute(
            f'SELECT {cols} FROM hourly WHERE StationId=? AND Datetime=? LIMIT 1',
            (station, prev)).fetchone()
    finally:
        con.close()
    return dict(zip(TARGETS, row)) if row else {}


def build_lag_features(station, when):
    """Best available lag-1 values: observed > station-hour mean > station mean > global mean."""
    prev_hour = (when.hour - 1) % 24
    layers = [
        ('observed', observed_previous_hour(station, when)),
        ('station-hour-average', station_hour_means.get((station, prev_hour), {})),
        ('station-average', station_means.get(station, {})),
        ('global-average', global_means),
    ]
    values, used = {}, []
    for t in TARGETS:
        for name, source in layers:
            v = source.get(t)
            if v is not None and not pd.isna(v):
                col = f'{t}_lag1'
                v = float(v)
                if col in clip_bounds:
                    lo, hi = clip_bounds[col]
                    v = min(max(v, lo), hi)
                values[col] = v
                if name not in used:
                    used.append(name)
                break
    return values, used


def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode('ascii')


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html', stations=station_list, targets=TARGETS)


@app.route('/predict', methods=['POST'])
def predict():
    station = request.form.get('station', '').strip()
    date_str = request.form.get('date', '').strip()
    hour_str = request.form.get('hour', '').strip()

    if station not in valid_stations:
        return error(f"Unknown station: {station!r}")
    try:
        hour = int(hour_str)
        if not 0 <= hour <= 23:
            raise ValueError
    except ValueError:
        return error('Hour must be an integer between 0 and 23')
    try:
        when = datetime.datetime.strptime(date_str, '%Y-%m-%d').replace(hour=hour)
    except ValueError:
        return error('Date must be in YYYY-MM-DD format (e.g., 2019-01-15)')

    try:
        lags, used = build_lag_features(station, when)
        row = {'StationId': [station], 'Year': [when.year], 'Month': [when.month],
               'Day': [when.day], 'Hour': [hour]}
        row.update({k: [v] for k, v in lags.items()})
        pred = model.predict(pd.DataFrame(row))[0]
        predictions = {t: float(v) for t, v in zip(TARGETS, pred)}
        return jsonify({
            'predictions': predictions,
            'AQI_Bucket': aqi_bucket(predictions['AQI']),
            'lag_source': used,
        })
    except Exception:
        logger.exception("Prediction error")
        return error('Prediction failed. See server log.', 500)


@app.route('/visualize', methods=['POST'])
def visualize():
    station = request.form.get('station', '').strip()
    params = request.form.getlist('parameters') or list(TARGETS)

    if station not in valid_stations:
        return error(f"Unknown station: {station!r}")
    bad = [p for p in params if p not in TARGETS]
    if bad:
        return error(f"Unknown parameters: {bad}")
    if station not in station_means or station not in daily_by_station:
        return error(f"No data available for station {station}", 404)

    try:
        averages = pd.Series(station_means[station])[params].dropna()
        positive = averages[averages > 0]
        daily = daily_by_station[station]

        # Pie
        fig = Figure(figsize=(7, 6))
        ax = fig.subplots()
        ax.pie(positive.values, labels=positive.index, autopct='%1.1f%%')
        ax.set_title('Distribution of Selected Gases')
        pie_url = fig_to_b64(fig)

        # Bar
        fig = Figure(figsize=(8, 5))
        ax = fig.subplots()
        ax.bar(averages.index, averages.values, color='#66b3ff')
        ax.set_title('Average Values of Selected Gases')
        ax.set_ylabel('Average value')
        ax.set_xlabel('Parameter')
        fig.autofmt_xdate()
        bar_url = fig_to_b64(fig)

        # Line
        fig = Figure(figsize=(10, 5))
        ax = fig.subplots()
        for p in params:
            ax.plot(daily.index, daily[p], label=p, linewidth=1)
        ax.set_title('Daily Trend of Selected Gases')
        ax.set_ylabel('Value')
        ax.set_xlabel('Date')
        ax.legend(loc='upper right', ncol=2, fontsize=8)
        fig.autofmt_xdate()
        line_url = fig_to_b64(fig)

        return jsonify({'pie_url': pie_url, 'bar_url': bar_url, 'line_url': line_url})
    except Exception:
        logger.exception("Visualization error")
        return error('Visualization failed. See server log.', 500)


if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1')