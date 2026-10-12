import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import joblib
import os
import time
import logging
import json
import numpy as np
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_model_svm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProgressMonitor:
    """Class to print training progress every 10-15 seconds."""
    def __init__(self, interval=10):
        self.interval = interval
        self.running = False
        self.start_time = None

    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._print_progress)
        self.thread.daemon = True
        self.thread.start()

    def _print_progress(self):
        while self.running:
            elapsed = time.time() - self.start_time
            logger.info(f"Training in progress... Elapsed time: {elapsed:.1f} seconds")
            time.sleep(self.interval + np.random.uniform(-5, 5))  # Randomize 10-15s

    def stop(self):
        self.running = False
        self.thread.join()

def load_data(file_path, dtype=None, sample_n=None):
    """Load a CSV file with optional sampling for large datasets."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path, dtype=dtype)
        if sample_n:
            sample_frac = min(sample_n / len(df), 1.0)
            df = df.sample(n=sample_n, random_state=42)
            logger.info(f"Sampled {sample_n} rows ({sample_frac*100:.2f}% of data)")
        logger.info(f"Successfully loaded {file_path} with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise

def validate_data(df, required_columns):
    """Validate that the DataFrame has all required columns."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    logger.info("All required columns present")

def preprocess_data(station_hour, targets):
    """Preprocess the data: parse datetime, extract features, add lags, handle missing values."""
    try:
        if 'Datetime' not in station_hour.columns:
            raise ValueError("Column 'Datetime' not found in station_hour data")
        station_hour['Datetime'] = pd.to_datetime(station_hour['Datetime'], errors='coerce')
        
        if station_hour['Datetime'].isna().any():
            logger.warning("Some Datetime values could not be parsed; filling with forward fill")
            station_hour['Datetime'].fillna(method='ffill', inplace=True)

        # Sort for lag features
        station_hour = station_hour.sort_values(['StationId', 'Datetime'])

        # Add lag features (previous hour)
        for target in targets:
            station_hour[f'{target}_lag1'] = station_hour.groupby('StationId')[target].shift(1)
            station_hour[f'{target}_lag1'] = station_hour[f'{target}_lag1'].fillna(station_hour[target].mean())

        station_hour['Year'] = station_hour['Datetime'].dt.year
        station_hour['Month'] = station_hour['Datetime'].dt.month
        station_hour['Day'] = station_hour['Datetime'].dt.day
        station_hour['Hour'] = station_hour['Datetime'].dt.hour

        data = station_hour.drop(['Datetime', 'AQI_Bucket'], axis=1, errors='ignore')
        
        numerics = data.select_dtypes(include=['float64', 'int64']).columns
        if not numerics.empty:
            for col in numerics:
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val)
                low, high = data[col].quantile([0.01, 0.99])
                data[col] = data[col].clip(low, high)
            logger.info("Filled missing numeric values with medians and clipped outliers")
        else:
            logger.warning("No numeric columns found for missing value imputation")

        logger.info(f"Processed data shape: {data.shape}")
        logger.info(f"Sample data:\n{data.head().to_string()}")
        return data
    except Exception as e:
        logger.error(f"Error in preprocessing data: {str(e)}")
        raise

def train_and_save_model(data, features, targets, model_path, metrics_path):
    """Train SVM with k-fold cross-validation, compute accuracy, and save."""
    try:
        if len(data) < 10:
            raise ValueError("Dataset too small for training (less than 10 rows)")

        X = data[features]
        y = data[targets]

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['StationId'])
            ],
            remainder='passthrough'
        )

        # Model pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', MultiOutputRegressor(SVR(kernel='linear', C=1.0, verbose=0, n_jobs=-1)))
        ])

        # K-fold cross-validation (5 folds)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores_per_fold = {target: [] for target in targets}
        
        logger.info("Starting k-fold cross-validation...")
        progress = ProgressMonitor(interval=10)
        progress.start()
        start_time = time.time()
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            for i, target in enumerate(targets):
                r2 = r2_score(y_test[target], y_pred[:, i])
                r2_scores_per_fold[target].append(r2)
            logger.info(f"Fold {fold+1} R² scores: { {k: v[-1] for k, v in r2_scores_per_fold.items()} }")
        
        progress.stop()
        # Compute mean R² scores across folds
        r2_scores = {target: np.mean(scores) for target, scores in r2_scores_per_fold.items()}
        logger.info(f"Mean R² scores across 5 folds: {r2_scores}")

        # Train final model on full data
        logger.info("Training final model on full data...")
        progress.start()
        model.fit(X, y)
        progress.stop()
        end_time = time.time()
        logger.info(f"Training completed in {end_time - start_time:.2f} seconds")

        # Save the model
        joblib.dump(model, model_path)
        logger.info(f"Model saved successfully to {model_path}")

        # Save accuracy metrics
        with open(metrics_path, 'w') as f:
            json.dump(r2_scores, f, indent=4)
        logger.info(f"Accuracy metrics saved to {metrics_path}")

    except Exception as e:
        logger.error(f"Error in training or saving model: {str(e)}")
        raise

def main():
    """Main function to orchestrate data loading, preprocessing, and model training."""
    try:
        os.makedirs('models', exist_ok=True)
        model_path = 'models/air_quality_model_svm.pkl'
        metrics_path = 'models/accuracy_metrics_svm.json'

        required_columns = [
            'StationId', 'Datetime', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 
            'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI'
        ]
        targets = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 
                   'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
        features = ['StationId', 'Year', 'Month', 'Day', 'Hour'] + [f'{t}_lag1' for t in targets]

        logger.info("Loading data...")
        stations = load_data('data/stations.csv')
        station_hour = load_data('data/station_hour_cleaned.csv', 
                               dtype={'AQI_Bucket': 'object'}, 
                               sample_n=500000)  # Sample 500k rows

        validate_data(station_hour, required_columns)

        logger.info("Preprocessing data...")
        data = preprocess_data(station_hour, targets)

        train_and_save_model(data, features, targets, model_path, metrics_path)

    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()