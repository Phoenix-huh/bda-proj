import os

import numpy as np
import pandas as pd
import pytest

from advisory import advisory_for, aqi_bucket
from covid import bh_adjust
from features import ALL_POLLUTANTS, build_station_frame
from modeling import conformal_qhat


def _synthetic(n=200):
    df = pd.DataFrame({'Datetime': pd.date_range('2019-01-01', periods=n, freq='h')})
    for p in ALL_POLLUTANTS:
        df[p] = np.arange(n, dtype=float)
    return df


def test_feature_alignment_has_no_leakage():
    f = build_station_frame(_synthetic(), horizons=[1, 24], targets=['AQI'])
    row = f.iloc[100]
    assert row['cur_AQI'] == 100
    assert row['lag6_AQI'] == 94
    assert row['lag24_AQI'] == 76
    assert row['y_AQI_h1'] == 101
    assert row['y_AQI_h24'] == 124
    assert np.isnan(f.iloc[-1]['y_AQI_h1'])


def test_gap_gives_nan_instead_of_wrong_lag():
    df = _synthetic().drop(index=50)            # remove one hour
    f = build_station_frame(df, horizons=[1], targets=['AQI'])
    assert np.isnan(f.iloc[50]['cur_AQI'])
    assert np.isnan(f.iloc[56]['lag6_AQI'])      # would be the missing hour


def test_aqi_buckets():
    assert aqi_bucket(50) == 'Good'
    assert aqi_bucket(51) == 'Satisfactory'
    assert aqi_bucket(250) == 'Poor'
    assert aqi_bucket(401) == 'Severe'


def test_advisory_has_cigarette_equivalent():
    a = advisory_for(180, 66)
    assert a['bucket'] == 'Moderate'
    assert a['cigarettes_per_day'] == 3.0
    assert advisory_for(180)['cigarettes_per_day'] is None


def test_conformal_quantile():
    q = conformal_qhat(np.arange(100), alpha=0.1)
    assert 90 <= q <= 92


def test_bh_adjust_monotone_and_bounded():
    p = np.array([0.001, 0.02, 0.03, 0.5, np.nan])
    adj = bh_adjust(p)
    assert np.isnan(adj[-1])
    assert np.all(adj[:4] >= p[:4]) and np.all(adj[:4] <= 1)


# ---- API tests (skipped when artifacts have not been generated yet) ----
@pytest.fixture
def client():
    from app import app
    app.config['TESTING'] = True
    return app.test_client()


def test_advisory_endpoint(client):
    r = client.get('/api/advisory?aqi=320&pm25=110')
    assert r.status_code == 200
    assert r.get_json()['bucket'] == 'Very Poor'
    assert client.get('/api/advisory?aqi=abc').status_code == 400


@pytest.mark.skipif(not os.path.exists('cache/station_daily.csv'), reason='run build_cache.py')
def test_map_endpoint(client):
    r = client.get('/api/map-data?layer=aqi&date=2019-11-03')
    assert r.status_code == 200
    assert 'points' in r.get_json()
    assert client.get('/api/map-data?layer=aqi&date=bad').status_code == 400


@pytest.mark.skipif(not os.path.exists('models/forecast_models.joblib'), reason='run train_forecast.py')
def test_forecast_endpoint(client):
    s = client.get('/api/station-range?station=AP001').get_json()
    date, hour = s['max'].split(' ')[0], int(s['max'].split(' ')[1][:2])
    r = client.post('/api/forecast', data={'station': 'AP001', 'date': date, 'hour': hour,
                                           'target': 'AQI', 'explain_horizon': 24})
    assert r.status_code == 200
    d = r.get_json()
    item = d['forecasts']['AQI'][0]
    assert item['lower'] <= item['median'] <= item['upper']
    assert len(d['explanation']['top']) == 8
    assert client.post('/api/forecast', data={'station': 'NOPE', 'date': date, 'hour': 0}).status_code == 400
