# Air Quality Prediction and Analytics Dashboard

An end-to-end machine learning web application for monitoring, predicting, and analyzing air quality across 110 monitoring stations throughout India.

---

## 1. Overview

This application predicts 13 air quality parameters along with the national Air Quality Index (AQI) bucket for any monitoring station, date, and hour. In addition, it generates interactive historical visualizations (pie charts, bar charts, and daily trend lines) for selected pollutants.

### Predicted Targets (13 parameters)
- **Particulate Matter:** PM2.5, PM10
- **Nitrogen Compounds:** NO, NO2, NOx, NH3
- **Carbon & Sulfur:** CO, SO2, O3
- **Volatile Organic Compounds (VOCs):** Benzene, Toluene, Xylene
- **Air Quality Index:** AQI (mapped to buckets: *Good*, *Satisfactory*, *Moderate*, *Poor*, *Very Poor*, *Severe*)

---

## 2. Setup

### Prerequisites
- Python 3.11 or 3.12
- Git with Git LFS (if cloning dataset files)

### Virtual Environment & Dependencies
```powershell
# Windows
py -3.12 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# Linux / macOS
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Data

The project uses the Kaggle **"Air Quality Data in India (2015-2020)"** dataset:
- `data/stations.csv`: Metadata for 110 monitoring stations (StationId, City, State, Status).
- `data/station_hour_cleaned.csv`: 2,589,083 hourly records across India.
- `data/city_hour.csv`: City-level hourly aggregates.

> **Note on Large Files**: If cloning fresh from a repository configured with Git LFS:
> ```bash
> git lfs pull
> ```

---

## 4. First-Time Build

Before launching the web application or running tests, generate the precomputed cache and train the production model:

```powershell
# 1. Build offline SQLite database and summary caches (~1.5 minutes)
python build_cache.py

# 2. Train the Ridge regression model with 5-fold cross-validation (~15 seconds)
python train_model.py
```

### Cache Generation Details
`build_cache.py` processes the 2.59M row dataset in chunked streams without loading the entire CSV into memory:
- `cache/hourly.db`: SQLite database indexing `(StationId, Datetime)` for O(1) hourly lag lookups.
- `cache/station_daily.csv`: Precomputed daily averages for rapid historical trend plotting.
- `cache/station_means.csv`: Station-level overall pollutant averages.
- `cache/station_hour_means.csv`: Station-level hour-of-day pollutant profiles.
- `cache/global_means.json`: Dataset-wide fallback averages.

---

## 5. Running the Application

```powershell
python app.py
```

Open your browser to:
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

### Configuration Options
- **Debug Mode**: Set `FLASK_DEBUG=1` to enable debug reloader:
  ```powershell
  $env:FLASK_DEBUG="1"; python app.py
  ```
- **Custom Model Path**: Override the model loaded at runtime:
  ```powershell
  $env:MODEL_PATH="models/air_quality_model.pkl"; python app.py
  ```

---

## 6. API Endpoints

| Endpoint | Method | Input Parameters | Response Format / Details |
|---|---|---|---|
| `/` | `GET` | None | Renders `templates/index.html` dashboard with dropdown stations and pollutant list. |
| `/predict` | `POST` | `station` (str, e.g. `'AP001'`)<br>`date` (YYYY-MM-DD, e.g. `'2019-10-14'`)<br>`hour` (int 0-23, e.g. `'12'`) | `200 OK` JSON:<br>`{ "predictions": { "PM2.5": ..., "AQI": ... }, "AQI_Bucket": "Moderate", "lag_source": ["observed"] }`<br>`400 Bad Request` on invalid input or station. |
| `/visualize` | `POST` | `station` (str, e.g. `'AP001'`)<br>`parameters` (optional list of pollutant names) | `200 OK` JSON:<br>`{ "pie_url": "...", "bar_url": "...", "line_url": "..." }` containing Base64 PNG images.<br>`400/404` on invalid stations or parameters. |

### Feature Engineering & Hierarchical Lag Fallback
To provide accurate predictions even when recent historical data has missing readings, the prediction pipeline uses a 4-tier fallback:
1. **`observed`**: Exact reading from SQLite `cache/hourly.db` at $T - 1\text{ hour}$.
2. **`station-hour-average`**: Typical pollutant level for this station at this exact hour of the day.
3. **`station-average`**: Overall historical average for this station.
4. **`global-average`**: Dataset-wide pollutant average.

---

## 7. Machine Learning Models & Performance

Models are trained using temporal lag-1 features, station one-hot encoding, and date components (`Year`, `Month`, `Day`, `Hour`). Numerical lag features are bounded using training-set 99.9th percentiles saved in `models/clip_bounds.json`.

### Model Evaluation ($R^2$ Scores across 5 Folds)

| Target Parameter | Ridge Regression (`models/accuracy_metrics.json`) | Baseline Linear (`models/accuracy_metrics_LG.json`, pre-lag) |
|---|:---:|:---:|
| **AQI** | **0.9579** | 0.8126 |
| **PM2.5** | **0.8791** | 0.6558 |
| **PM10** | **0.8725** | 0.6482 |
| **NO** | 0.7537 | 0.5218 |
| **NO2** | 0.8499 | 0.6274 |
| **NOx** | 0.8029 | 0.5693 |
| **NH3** | 0.8606 | 0.6078 |
| **CO** | 0.7731 | 0.5489 |
| **SO2** | 0.7504 | 0.5142 |
| **O3** | 0.8135 | 0.5821 |
| **Benzene** | 0.8179 | 0.6035 |
| **Toluene** | 0.8408 | 0.6329 |
| **Xylene** | 0.8223 | 0.5982 |

- **Ridge Regression (Production default)**: MultiOutput Ridge regression ($a=1.0$) trained with 5-fold cross validation. Achieves $0.958\ R^2$ on AQI and $0.879\ R^2$ on PM2.5.
- **LinearSVR (Optional comparison)**: Can be trained with `python train_model.py --model linearsvr`. Note that SVR requires significantly longer training time due to liblinear multi-target fits.
- **Pre-lag baseline (`accuracy_metrics_LG.json`)**: Preserved historical record of the original linear regression model prior to lag-1 feature engineering.

---

## 8. Testing

Run automated unit and integration tests using pytest:

```powershell
pytest -v
```

Tests cover:
- Home route rendering (`/`)
- Prediction with valid stations and historical dates (`observed` lag verification)
- Prediction with future dates (`fallback` lag verification)
- Input validation: invalid station IDs, invalid date formats, out-of-range hours, empty requests
- Visualization generation: default parameters, custom parameter subsets, error handling for unknown stations

---

## 9. Known Limitations

1. **Future Dates are Not Recursive Forecasts**: The dataset covers 2015 to 2020. For queries beyond the dataset (e.g. 2026), observed readings are unavailable, so the model gracefully falls back to typical station-hour profiles. It reflects seasonal and diurnal expectations rather than true multi-step weather forecasts.
2. **Lag-1 Consecutive Assumption**: Gaps in raw recording stations can mean consecutive rows have time gaps. The SQLite query explicitly searches for $(T - 1\text{ hour})$ to ensure temporal accuracy.
3. **Linearity**: Ridge regression assumes linear relationships with bounded features. Complex meteorological interactions or non-linear atmospheric chemistry are better captured with gradient boosting models (e.g., LightGBM).
4. **Independent AQI Prediction**: AQI is predicted directly as a regression target alongside individual gases rather than calculated mathematically via CPCB sub-index breakpoints.
5. **Development Server**: The built-in Flask development server is intended for local evaluation. For production deployments, deploy behind a WSGI server such as Waitress (Windows) or Gunicorn (Linux) with a reverse proxy.
