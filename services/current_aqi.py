"""Service to fetch current air quality measurements from the OpenAQ API.

Never hard-codes API keys: reads OPENAQ_API_KEY from environment.
Caches responses per city for 30 minutes.
Maps pollutant levels to CPCB AQI categories reusing advisory.py.
"""
import csv
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from advisory import COLORS, aqi_bucket

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COORDS_PATH = os.path.join(BASE_DIR, 'data', 'city_coords.csv')
CACHE_TTL = 30 * 60  # 30 minutes in seconds

# In-memory response cache: {city_lower: (timestamp, result_dict)}
_cache = {}


def _load_city_coords():
    coords = {
        'delhi': (28.6139, 77.2090),
        'kolkata': (22.5726, 88.3639),
        'mumbai': (19.0760, 72.8777),
        'bengaluru': (12.9716, 77.5946),
        'chennai': (13.0827, 80.2707),
        'hyderabad': (17.3850, 78.4867),
        'ahmedabad': (23.0225, 72.5714),
        'pune': (18.5204, 73.8567),
        'jaipur': (26.9124, 75.7873),
        'lucknow': (26.8467, 80.9462),
        'patna': (25.5941, 85.1376),
    }
    if os.path.exists(COORDS_PATH):
        try:
            with open(COORDS_PATH, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    city = row.get('City', '').strip().lower()
                    try:
                        coords[city] = (float(row['Lat']), float(row['Lon']))
                    except (ValueError, KeyError):
                        continue
        except Exception as e:
            logger.warning(f"Could not load city_coords.csv: {e}")
    return coords


CITY_COORDS = _load_city_coords()


def clear_cache():
    """Clear cached responses (used in tests or manual refresh)."""
    _cache.clear()


def _calculate_sub_index(conc, breakpoints):
    """Linear interpolation for CPCB AQI sub-index."""
    if conc is None or conc < 0:
        return None
    for bp in breakpoints:
        c_low, c_high, i_low, i_high = bp
        if c_low <= conc <= c_high:
            return i_low + ((i_high - i_low) / (c_high - c_low)) * (conc - c_low)
    # If concentration exceeds highest breakpoint, extrapolate from the top band
    c_low, c_high, i_low, i_high = breakpoints[-1]
    return i_high + ((i_high - i_low) / (c_high - c_low)) * (conc - c_high)


def calculate_cpcb_aqi(pm25=None, pm10=None, no2=None):
    """Compute approximate CPCB AQI and category from available criteria pollutants."""
    # Breakpoint tables: (C_low, C_high, I_low, I_high)
    pm25_bp = [
        (0, 30, 0, 50),
        (30, 60, 50, 100),
        (60, 90, 100, 200),
        (90, 120, 200, 300),
        (120, 250, 300, 400),
        (250, 500, 400, 500),
    ]
    pm10_bp = [
        (0, 50, 0, 50),
        (50, 100, 50, 100),
        (100, 250, 100, 200),
        (250, 350, 200, 300),
        (350, 430, 300, 400),
        (430, 500, 400, 500),
    ]
    no2_bp = [
        (0, 40, 0, 50),
        (40, 80, 50, 100),
        (80, 180, 100, 200),
        (180, 280, 200, 300),
        (280, 400, 300, 400),
        (400, 500, 400, 500),
    ]

    sub_indices = []
    if pm25 is not None:
        idx = _calculate_sub_index(pm25, pm25_bp)
        if idx is not None:
            sub_indices.append(idx)
    if pm10 is not None:
        idx = _calculate_sub_index(pm10, pm10_bp)
        if idx is not None:
            sub_indices.append(idx)
    if no2 is not None:
        idx = _calculate_sub_index(no2, no2_bp)
        if idx is not None:
            sub_indices.append(idx)

    if not sub_indices:
        return None, 'Unknown', '#94a3b8'

    overall_aqi = max(sub_indices)
    category = aqi_bucket(overall_aqi)
    color = COLORS.get(category, '#94a3b8')
    return round(overall_aqi), category, color


def _make_unavailable_response(city, error_msg='live data unavailable'):
    return {
        'available': False,
        'city': city,
        'error': error_msg,
        'note': 'Live data is unvalidated and can differ from CPCB archives.',
    }


def get_current_aqi(city='Delhi', force_refresh=False):
    """Fetch current air quality for a city from OpenAQ API with 30-minute caching.

    Returns a dict with status, pollutant values, timestamp, and CPCB category.
    """
    city_norm = (city or 'Delhi').strip()
    cache_key = city_norm.lower()

    now = time.time()
    if not force_refresh and cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if now - cached_time < CACHE_TTL:
            return cached_data

    api_key = os.environ.get('OPENAQ_API_KEY', '').strip()
    if not api_key:
        logger.info("OPENAQ_API_KEY environment variable not set; returning unavailable.")
        res = _make_unavailable_response(city_norm, 'live data unavailable (OPENAQ_API_KEY not configured)')
        _cache[cache_key] = (now, res)
        return res

    coords = CITY_COORDS.get(cache_key)
    if not coords:
        # Fallback to Delhi if unrecognized
        logger.warning(f"No coordinates found for city '{city_norm}'")
        res = _make_unavailable_response(city_norm, f"live data unavailable (unknown coordinates for '{city_norm}')")
        _cache[cache_key] = (now, res)
        return res

    lat, lon = coords
    headers = {
        'X-API-Key': api_key,
        'User-Agent': 'AQI-India-Trends/1.0',
        'Accept': 'application/json',
    }

    try:
        # 1. Query OpenAQ v3 locations near the city coordinates
        loc_url = (
            f"https://api.openaq.org/v3/locations?"
            f"coordinates={lat:.4f},{lon:.4f}&radius=25000&limit=5"
        )
        req = urllib.request.Request(loc_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            loc_data = json.loads(resp.read().decode('utf-8'))

        results = loc_data.get('results', [])
        if not results:
            res = _make_unavailable_response(city_norm, 'live data unavailable (no stations found near city)')
            _cache[cache_key] = (now, res)
            return res

        # 2. Query latest measurements for the top location(s)
        # Try up to 3 locations in case the first has no active sensors
        best_readings = {'pm25': None, 'pm10': None, 'no2': None}
        station_name = results[0].get('name', f"{city_norm} station")
        latest_timestamp = None

        for loc in results[:3]:
            loc_id = loc.get('id')
            if not loc_id:
                continue
            latest_url = f"https://api.openaq.org/v3/locations/{loc_id}/latest"
            try:
                lreq = urllib.request.Request(latest_url, headers=headers)
                with urllib.request.urlopen(lreq, timeout=8) as lresp:
                    latest_data = json.loads(lresp.read().decode('utf-8'))
                for item in latest_data.get('results', []):
                    param_info = item.get('parameter', {})
                    param_name = str(param_info.get('name', '')).lower().replace('.', '')
                    val = item.get('value')
                    dt_info = item.get('datetime', {})
                    dt_str = dt_info.get('local') or dt_info.get('utc')

                    if param_name in ('pm25', 'pm25') and best_readings['pm25'] is None and val is not None:
                        best_readings['pm25'] = round(float(val), 1)
                        if not latest_timestamp:
                            latest_timestamp = dt_str
                    elif param_name == 'pm10' and best_readings['pm10'] is None and val is not None:
                        best_readings['pm10'] = round(float(val), 1)
                        if not latest_timestamp:
                            latest_timestamp = dt_str
                    elif param_name == 'no2' and best_readings['no2'] is None and val is not None:
                        best_readings['no2'] = round(float(val), 1)
                        if not latest_timestamp:
                            latest_timestamp = dt_str
            except Exception as e:
                logger.debug(f"Failed fetching latest for location {loc_id}: {e}")
                continue

            # If we found at least PM2.5 or PM10, use this station
            if best_readings['pm25'] is not None or best_readings['pm10'] is not None:
                station_name = loc.get('name', station_name)
                break

        if all(v is None for v in best_readings.values()):
            res = _make_unavailable_response(city_norm, 'live data unavailable (no recent pollutant readings)')
            _cache[cache_key] = (now, res)
            return res

        aqi, category, color = calculate_cpcb_aqi(
            pm25=best_readings['pm25'],
            pm10=best_readings['pm10'],
            no2=best_readings['no2'],
        )

        res = {
            'available': True,
            'city': city_norm,
            'station_name': station_name,
            'pm25': best_readings['pm25'],
            'pm10': best_readings['pm10'],
            'no2': best_readings['no2'],
            'aqi': aqi,
            'category': category,
            'color': color,
            'timestamp': latest_timestamp or 'Just now',
            'source': 'OpenAQ API',
            'note': 'Live data is unvalidated and can differ from CPCB archives.',
        }
        _cache[cache_key] = (now, res)
        return res

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning(f"OpenAQ API request failed for {city_norm}: {e}")
        res = _make_unavailable_response(city_norm, 'live data unavailable')
        _cache[cache_key] = (now, res)
        return res
    except Exception as e:
        logger.exception(f"Unexpected error fetching OpenAQ data for {city_norm}: {e}")
        res = _make_unavailable_response(city_norm, 'live data unavailable')
        _cache[cache_key] = (now, res)
        return res

