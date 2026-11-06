import json
import os
import unittest.mock as mock
import numpy as np
import pandas as pd
import pytest

from app import app
from services.current_aqi import (
    calculate_cpcb_aqi,
    clear_cache,
    get_current_aqi,
)
from trends import (
    calculate_pct_change_vs_2020,
    get_trends_data,
    load_published_trends,
)
from scripts.build_trends import clean_dataframe, run_trend_analysis


@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()


# ── 1. Unit Tests: Percent Change vs 2020 ──────────────────────────────────────
def test_calculate_pct_change_delhi_values():
    delhi_values = {
        "2018": 115.8,
        "2019": 109.2,
        "2020": 95.1,
        "2021": 106.2,
        "2022": 98.6,
        "2023": 100.9,
        "2024": 104.7,
        "2025": 100.0,
    }
    pct = calculate_pct_change_vs_2020(delhi_values, '2020')
    assert pct["2020"] == 0.0
    # ((115.8 - 95.1) / 95.1) * 100 = 21.7665... -> 21.77
    assert pct["2018"] == 21.77
    # ((109.2 - 95.1) / 95.1) * 100 = 14.826... -> 14.83
    assert pct["2019"] == 14.83
    # ((106.2 - 95.1) / 95.1) * 100 = 11.6719... -> 11.67
    assert pct["2021"] == 11.67
    assert pct["2025"] == 5.15


def test_calculate_pct_change_kolkata_values():
    kolkata_values = {"2019": 41.8, "2020": 27.9, "2021": 37.3}
    pct = calculate_pct_change_vs_2020(kolkata_values, '2020')
    assert pct["2020"] == 0.0
    # ((41.8 - 27.9) / 27.9) * 100 = 49.8207... -> 49.82
    assert pct["2019"] == 49.82
    # ((37.3 - 27.9) / 27.9) * 100 = 33.6917... -> 33.69
    assert pct["2021"] == 33.69


def test_calculate_pct_change_edge_cases():
    assert calculate_pct_change_vs_2020({}) == {}
    assert calculate_pct_change_vs_2020({"2021": 50}, base_year='2020') == {}
    assert calculate_pct_change_vs_2020({"2020": 0, "2021": 50}) == {}
    assert calculate_pct_change_vs_2020({"2020": "bad", "2021": 50}) == {}


# ── 2. Published Trends Data ──────────────────────────────────────────────────
def test_published_trends_file_and_structure():
    data = load_published_trends()
    assert 'delhi_annual_pm25' in data
    assert 'kolkata_marmay_pm25' in data

    delhi = data['delhi_annual_pm25']
    assert delhi['unit'] == 'ug/m3'
    assert delhi['values']['2020'] == 95.1
    assert 'source' in delhi
    assert 'pct_change_vs_2020' in delhi
    assert delhi['pct_change_vs_2020']['2020'] == 0.0

    kolkata = data['kolkata_marmay_pm25']
    assert kolkata['unit'] == 'ug/m3'
    assert kolkata['values']['2020'] == 27.9
    assert 'source' in kolkata
    assert 'pct_change_vs_2020' in kolkata


# ── 3. Routes: /trends, /api/trends, /api/current ─────────────────────────────
def test_trends_page_renders(client):
    res = client.get('/trends')
    assert res.status_code == 200
    assert b"Trends: 2020 vs Later Years" in res.data
    assert b"Current Air Quality" in res.data
    assert b"Sources use different station sets and are not directly comparable" in res.data
    assert b"chart-a-median-aqi" in res.data
    assert b"chart-b-pct-change" in res.data
    assert b"chart-c-category-shares" in res.data
    assert b"chart-d-named-cities" in res.data
    assert b"chart-e-pre-vs-window" in res.data
    assert b"fixed panel of 58 cities" in res.data
    assert b"2022 window was the highest" in res.data


def test_api_trends_endpoint(client):
    res = client.get('/api/trends')
    assert res.status_code == 200
    data = res.get_json()
    assert 'delhi_annual_pm25' in data
    assert 'kolkata_marmay_pm25' in data
    assert 'pct_change_vs_2020' in data['delhi_annual_pm25']
    assert data['delhi_annual_pm25']['pct_change_vs_2020']['2020'] == 0.0
    assert data['kolkata_marmay_pm25']['pct_change_vs_2020']['2020'] == 0.0


def test_api_aqi_trends_endpoint(client):
    res = client.get('/api/aqi-trends')
    assert res.status_code == 200
    data = res.get_json()
    assert data['panel_size'] == 58
    assert len(data['panel_cities']) == 58
    assert 'median_window_aqi' in data
    assert 'pct_change_vs_2020' in data
    assert 'category_shares' in data
    assert 'named_cities_window_means' in data
    assert 'pre_window_median_aqi' in data

    # Sanity checks on values
    assert abs(data['median_window_aqi']['2020'] - 89.4) < 1.0
    assert abs(data['median_window_aqi']['2022'] - 138.5) < 1.0
    assert abs(data['pct_change_vs_2020']['2021']['median_pct_change'] - 31.9) < 1.0
    assert abs(data['pct_change_vs_2020']['2022']['median_pct_change'] - 58.5) < 1.0

    # Named cities check
    for c in ['Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Bengaluru', 'Hyderabad', 'Lucknow', 'Ahmedabad']:
        assert c in data['named_cities_window_means']


# ── 6. Unit Tests: Panel Selection, Seeded Bootstrap & Analysis ─────────────────
def test_compute_fixed_panel_selection():
    from scripts.build_aqi_trends import compute_fixed_panel

    dates = pd.date_range('2020-03-25', '2025-05-31', freq='D')
    rows = []
    # CityA: present every day in window (68 days each year) -> >= 54.4 in all years -> in panel
    # CityB: present only 40 days in window in 2022 (< 54.4) -> not in panel
    for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
        window_dates = [d for d in pd.date_range(f'{yr}-03-25', f'{yr}-05-31', freq='D')]
        for d in window_dates:
            rows.append({'Date': d, 'City': 'CityA', 'AQI Value': 100.0, 'Air Quality': 'Moderate'})
        # CityB missing days in 2022
        b_dates = window_dates[:40] if yr == 2022 else window_dates
        for d in b_dates:
            rows.append({'Date': d, 'City': 'CityB', 'AQI Value': 110.0, 'Air Quality': 'Moderate'})

    df = pd.DataFrame(rows)
    panel, _ = compute_fixed_panel(df, years=[2020, 2021, 2022, 2023, 2024, 2025])
    assert 'CityA' in panel
    assert 'CityB' not in panel
    assert len(panel) == 1


def test_seeded_bootstrap_ci():
    from scripts.build_aqi_trends import compute_bootstrap_ci

    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    lo1, hi1 = compute_bootstrap_ci(arr, n_boot=5000, seed=0)
    lo2, hi2 = compute_bootstrap_ci(arr, n_boot=5000, seed=0)

    # Deterministic with seed=0
    assert lo1 == lo2
    assert hi1 == hi2
    assert lo1 <= 35.0 <= hi1



def test_api_current_graceful_failure_without_key(client, monkeypatch):
    clear_cache()
    monkeypatch.delenv('OPENAQ_API_KEY', raising=False)
    res = client.get('/api/current?city=Delhi')
    assert res.status_code == 200
    data = res.get_json()
    assert data['available'] is False
    assert 'live data unavailable' in data['error']
    assert 'note' in data


def test_api_current_mock_success(client, monkeypatch):
    clear_cache()
    mock_result = {
        'available': True,
        'city': 'Delhi',
        'station_name': 'Anand Vihar, Delhi',
        'pm25': 85.0,
        'pm10': 160.0,
        'no2': 45.0,
        'aqi': 183,
        'category': 'Moderate',
        'color': '#ffd400',
        'timestamp': '2026-11-06T08:30:00+05:30',
        'source': 'OpenAQ API',
        'note': 'Live data is unvalidated and can differ from CPCB archives.',
    }
    with mock.patch('services.current_aqi.get_current_aqi', return_value=mock_result):
        res = client.get('/api/current?city=Delhi')
        assert res.status_code == 200
        data = res.get_json()
        assert data['available'] is True
        assert data['pm25'] == 85.0
        assert data['category'] == 'Moderate'


# ── 4. CPCB Sub-Index Calculation & Cache ─────────────────────────────────────
def test_calculate_cpcb_aqi():
    # PM2.5 = 25 (0-30 -> Good)
    aqi, cat, col = calculate_cpcb_aqi(pm25=25)
    assert aqi <= 50
    assert cat == 'Good'

    # PM2.5 = 100 (91-120 -> Poor)
    aqi, cat, col = calculate_cpcb_aqi(pm25=100)
    assert 200 < aqi <= 300
    assert cat == 'Poor'

    # All None
    aqi, cat, col = calculate_cpcb_aqi()
    assert aqi is None
    assert cat == 'Unknown'


def test_current_aqi_caching(monkeypatch):
    clear_cache()
    monkeypatch.setenv('OPENAQ_API_KEY', 'dummy_key')

    # Mock openaq responses
    loc_payload = json.dumps({'results': [{'id': 999, 'name': 'Mock Station'}]}).encode('utf-8')
    meas_payload = json.dumps({
        'results': [
            {'parameter': {'name': 'pm25'}, 'value': 42.0, 'datetime': {'local': '2026-11-06T10:00:00'}},
            {'parameter': {'name': 'pm10'}, 'value': 75.0, 'datetime': {'local': '2026-11-06T10:00:00'}},
            {'parameter': {'name': 'no2'}, 'value': 30.0, 'datetime': {'local': '2026-11-06T10:00:00'}},
        ]
    }).encode('utf-8')

    mock_urlopen = mock.MagicMock()
    mock_resp1 = mock.MagicMock()
    mock_resp1.read.return_value = loc_payload
    mock_resp1.__enter__.return_value = mock_resp1

    mock_resp2 = mock.MagicMock()
    mock_resp2.read.return_value = meas_payload
    mock_resp2.__enter__.return_value = mock_resp2

    mock_urlopen.side_effect = [mock_resp1, mock_resp2]

    with mock.patch('urllib.request.urlopen', mock_urlopen):
        res1 = get_current_aqi('Delhi')
        assert res1['available'] is True
        assert res1['pm25'] == 42.0
        assert mock_urlopen.call_count == 2

        # Second call within 30 min should hit cache, not call urlopen again
        res2 = get_current_aqi('Delhi')
        assert res2['available'] is True
        assert mock_urlopen.call_count == 2  # no additional calls


# ── 5. build_trends Data Cleaning & Processing ────────────────────────────────
def test_clean_dataframe_removes_invalid_entries():
    raw = pd.DataFrame({
        'station': ['S1', 'S1', 'S2', 'S2', 'S3'],
        'city': ['Delhi', 'Delhi', 'Delhi', 'Kolkata', 'Kolkata'],
        'datetime': ['2020-04-01 10:00', '2020-04-01 10:00', '2020-04-02 10:00', '2020-04-01 10:00', '2020-04-01 10:00'],
        'pm2.5': [50.0, 50.0, -10.0, 999.0, 0.0],
        'pm10': [100.0, 100.0, 120.0, 0.0, 80.0],
        'no2': [30.0, 30.0, 35.0, 40.0, 25.0],
    })
    cleaned, pols = clean_dataframe(raw)
    assert 'PM2.5' in pols
    # duplicate (S1, 2020-04-01 10:00) should be dropped
    assert len(cleaned) == 4
    # negative should be NaN
    assert np.isnan(cleaned.loc[cleaned['Datetime'] == '2020-04-02 10:00', 'PM2.5'].iloc[0])
    # 999 placeholder should be NaN
    assert np.isnan(cleaned.loc[(cleaned['City'] == 'Kolkata') & (cleaned['StationId'] == 'S2'), 'PM2.5'].iloc[0])
    # 0.0 for PM2.5 should be NaN
    assert np.isnan(cleaned.loc[(cleaned['City'] == 'Kolkata') & (cleaned['StationId'] == 'S3'), 'PM2.5'].iloc[0])


def test_run_trend_analysis_synthetic():
    # Create synthetic lockdown window data for 2017-2022
    records = []
    cities = ['Delhi', 'Kolkata', 'Mumbai', 'Chennai', 'Bengaluru']
    for yr in [2018, 2019, 2020, 2021, 2022]:
        for m, d in [(4, 1), (4, 15), (5, 1), (5, 15)]:  # sample days in window
            for c_idx, c in enumerate(cities):
                base_pm = 50 + c_idx * 10
                # Simulate dip in 2020 and rebound
                val = base_pm * (0.6 if yr == 2020 else (0.9 if yr == 2021 else 1.0))
                records.append({
                    'StationId': f'{c}_1',
                    'City': c,
                    'Date': pd.Timestamp(f'{yr}-{m:02d}-{d:02d}'),
                    'PM2.5': val,
                    'PM10': val * 1.8,
                    'NO2': val * 0.4,
                })
    df = pd.DataFrame(records)
    res = run_trend_analysis(df, ['PM2.5', 'PM10'], min_days=2)
    assert 2021 in res['years']
    assert 2022 in res['years']
    assert 'PM2.5' in res['vs_2020']
    vs_2020_pm25 = res['vs_2020']['PM2.5']
    assert len(vs_2020_pm25) >= 1
    # 2021 should show positive rebound compared to 2020 (val was 0.6 in 2020, 0.9 in 2021 -> +50%)
    rec_2021 = next(r for r in vs_2020_pm25 if r['year'] == 2021)
    assert rec_2021['median_pct_change'] > 0
    assert rec_2021['n_cities'] == 5
    assert 'p_adj_BH' in rec_2021


# ── 6. COVID page extension integration tests ─────────────────────────────────
def test_covid_page_extension_heading_and_status(client):
    res = client.get('/covid')
    assert res.status_code == 200
    assert b"After the lockdown: 2020 vs 2021-2025 (AQI extension)" in res.data
    assert b"Original analysis: CPCB 2015-2020, PM2.5/PM10/NO2/CO/O3" in res.data
    assert b"Extension: daily AQI only, Kaggle krishnandansha/aqi-india-2020-to-2025-october" in res.data
    assert b"The original analysis uses data ending 1 July 2020; see the extension section below for 2021-2025." in res.data
    assert b"covid-chart-a-median-aqi" in res.data
    assert b"covid-chart-b-pct-change" in res.data
    assert b"covid-chart-c-category-shares" in res.data
    assert b"covid-chart-e-pre-vs-window" in res.data
    assert b"View City-Level Trends (Chart D) on the Trends Page" in res.data


def test_covid_page_original_lockdown_values_preserved(client):
    res = client.get('/covid')
    assert res.status_code == 200
    # Original lockdown table values (AQI -28.7)
    assert b"-28.7" in res.data
    assert b"COVID-19 Lockdown Impact" in res.data
    assert b"ML counterfactual" in res.data


def test_covid_page_extension_hidden_when_json_missing(client, monkeypatch):
    real_exists = os.path.exists
    def mock_exists(path):
        if 'aqi_trends.json' in str(path):
            return False
        return real_exists(path)
    monkeypatch.setattr('os.path.exists', mock_exists)
    monkeypatch.setattr('novelties.os.path.exists', mock_exists)
    res = client.get('/covid')
    assert res.status_code == 200
    assert b"Extension data not available" in res.data
    assert b"After the lockdown: 2020 vs 2021-2025 (AQI extension)" not in res.data
    # Original content still present
    assert b"-28.7" in res.data


