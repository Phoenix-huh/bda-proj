import pytest

from app import app

VALID_BUCKETS = {'Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'}


@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()


def test_home_renders(client):
    r = client.get('/')
    assert r.status_code == 200
    assert b'Air Quality Dashboard' in r.data


def test_predict_valid(client):
    r = client.post('/predict', data={'station': 'AP001', 'date': '2019-10-14', 'hour': '12'})
    assert r.status_code == 200
    data = r.get_json()
    assert data['AQI_Bucket'] in VALID_BUCKETS
    assert 'PM2.5' in data['predictions']
    assert len(data['predictions']) == 13
    assert data['lag_source']


def test_predict_future_date_uses_fallback(client):
    r = client.post('/predict', data={'station': 'AP001', 'date': '2030-01-01', 'hour': '0'})
    assert r.status_code == 200
    assert 'observed' not in r.get_json()['lag_source']


@pytest.mark.parametrize('payload', [
    {'station': 'NOPE', 'date': '2019-10-14', 'hour': '12'},
    {'station': 'AP001', 'date': '14-10-2019', 'hour': '12'},
    {'station': 'AP001', 'date': '2019-10-14', 'hour': '24'},
    {'station': 'AP001', 'date': '2019-10-14', 'hour': 'abc'},
    {},
])
def test_predict_invalid(client, payload):
    r = client.post('/predict', data=payload)
    assert r.status_code == 400
    assert 'error' in r.get_json()


def test_visualize_valid(client):
    r = client.post('/visualize', data={'station': 'AP001', 'parameters': ['PM2.5', 'PM10']})
    assert r.status_code == 200
    data = r.get_json()
    assert set(data) == {'pie_url', 'bar_url', 'line_url'}
    assert all(len(v) > 100 for v in data.values())


def test_visualize_defaults_to_all_parameters(client):
    r = client.post('/visualize', data={'station': 'AP001'})
    assert r.status_code == 200


def test_visualize_invalid(client):
    assert client.post('/visualize', data={'station': 'NOPE'}).status_code == 400
    r = client.post('/visualize', data={'station': 'AP001', 'parameters': ['FAKE']})
    assert r.status_code == 400
