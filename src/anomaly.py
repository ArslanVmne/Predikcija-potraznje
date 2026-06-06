import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(history: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Detect demand anomalies using IQR + Isolation Forest.
    Input: DataFrame with 'date' and 'sales' columns.
    Returns: same rows with added 'is_anomaly', 'direction', 'zscore' columns.
    """
    df = history.copy().sort_values("date").reset_index(drop=True)

    if len(df) < 14:
        df["is_anomaly"] = False
        df["direction"] = "normal"
        df["zscore"] = 0.0
        return df

    # Rolling stats for residual calculation
    roll_mean = df["sales"].rolling(7, min_periods=3).mean().bfill()
    roll_std = df["sales"].rolling(7, min_periods=3).std().bfill().replace(0, 1)
    df["_residual"] = (df["sales"] - roll_mean) / roll_std

    # IQR flag
    q1, q3 = df["_residual"].quantile(0.25), df["_residual"].quantile(0.75)
    iqr = q3 - q1
    iqr_flag = (df["_residual"] < q1 - 2.5 * iqr) | (df["_residual"] > q3 + 2.5 * iqr)

    # Isolation Forest flag
    features = np.column_stack([
        df["sales"].values,
        roll_mean.values,
        df["_residual"].values,
    ])
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    iso_labels = iso.fit_predict(features)
    iso_flag = iso_labels == -1

    df["is_anomaly"] = iqr_flag | iso_flag
    df["zscore"] = df["_residual"].round(2)
    df["direction"] = np.where(df["_residual"] > 0, "spike", "drop")
    df.drop(columns=["_residual"], inplace=True)

    return df
