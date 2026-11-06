"""Trends analysis and published data processing.

Computes percent change vs 2020 per year for published and own trends datasets.
"""
import json
import logging
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLISHED_TRENDS_PATH = os.path.join(BASE_DIR, 'data', 'published_trends.json')
OWN_TRENDS_PATH = os.path.join(BASE_DIR, 'data', 'own_trends.json')

logger = logging.getLogger(__name__)


def calculate_pct_change_vs_2020(values, base_year='2020'):
    """Calculate percent change vs base_year for a dict of {year_str: numeric_value}.

    Formula: ((value - base_value) / base_value) * 100
    Returns: dict of {year_str: round(pct_change, 2)}
    """
    if not values or str(base_year) not in values:
        return {}
    try:
        base_val = float(values[str(base_year)])
    except (ValueError, TypeError):
        return {}
    if base_val <= 0:
        return {}

    pct_changes = {}
    for year, val in values.items():
        try:
            v = float(val)
            pct = ((v - base_val) / base_val) * 100.0
            pct_changes[str(year)] = round(pct, 2)
        except (ValueError, TypeError):
            continue
    return pct_changes


def load_published_trends():
    """Load data/published_trends.json and calculate percent change vs 2020."""
    if not os.path.exists(PUBLISHED_TRENDS_PATH):
        return {}
    with open(PUBLISHED_TRENDS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for key, series in data.items():
        values = series.get('values', {})
        series['pct_change_vs_2020'] = calculate_pct_change_vs_2020(values, '2020')

    return data


def load_own_trends():
    """Load data/own_trends.json if present, else None."""
    if not os.path.exists(OWN_TRENDS_PATH):
        return None
    try:
        with open(OWN_TRENDS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading {OWN_TRENDS_PATH}: {e}")
        return None


def get_trends_data():
    """Build complete JSON response for /api/trends."""
    published = load_published_trends()
    own = load_own_trends()
    return {
        'published': published,
        'delhi_annual_pm25': published.get('delhi_annual_pm25'),
        'kolkata_marmay_pm25': published.get('kolkata_marmay_pm25'),
        'own_trends': own,
    }

