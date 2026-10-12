"""Model factory, metrics and conformal helper shared by the experiments."""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR

from features import feature_columns

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None


def _linear_pre():
    num = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['StationId']),
        ('num', num, feature_columns()),
    ])


def make_model(name, quantile=None, n_estimators=300):
    """Return an unfitted single-output regressor. X must be a DataFrame with
    a categorical 'StationId' column (see features.FEATURES)."""
    if name == 'ridge':
        return Pipeline([('pre', _linear_pre()), ('reg', Ridge(alpha=10.0))])
    if name == 'linearsvr':
        reg = TransformedTargetRegressor(
            regressor=LinearSVR(epsilon=0.1, C=1.0, dual='auto', max_iter=5000, random_state=42),
            transformer=StandardScaler())
        return Pipeline([('pre', _linear_pre()), ('reg', reg)])
    if name == 'hgb':
        return HistGradientBoostingRegressor(
            max_iter=n_estimators, learning_rate=0.08, max_leaf_nodes=31,
            categorical_features='from_dtype', random_state=42)
    if name == 'lightgbm':
        if LGBMRegressor is None:
            raise ImportError("pip install lightgbm")
        params = dict(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      min_child_samples=40, random_state=42, verbose=-1, n_jobs=-1)
        if quantile is not None:
            params.update(objective='quantile', alpha=quantile)
        return LGBMRegressor(**params)
    raise ValueError(f"Unknown model: {name}")


def regression_metrics(y_true, y_pred):
    return {
        'n': int(len(y_true)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
    }


def conformal_qhat(scores, alpha):
    """Split-conformal quantile with finite-sample correction."""
    scores = np.asarray(scores, float)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method='higher'))
