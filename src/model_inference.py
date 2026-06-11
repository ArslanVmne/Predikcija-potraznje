import numpy as np
import pandas as pd

from src.data_loader import (
    load_lgbm,
    load_lgbm_features,
    load_train,
    load_val_features,
    load_val_preds,
)


def _log_to_linear(x: np.ndarray) -> np.ndarray:
    """Convert LGBM log-space output (sales_log) to linear sales."""
    return np.expm1(np.clip(x, 0, 20))


def get_history(store: int, family: str, days: int = 90) -> pd.DataFrame:
    train = load_train()
    mask = (train["store_nbr"] == store) & (train["family"] == family)
    df = train[mask].sort_values("date").tail(days)[["date", "sales"]].copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def get_forecast(store: int, family: str) -> pd.DataFrame:
    """Returns val period forecast with CI band (±1.5×MAE).
    ensemble_pred is already in linear space (back-transformed in notebook)."""
    preds = load_val_preds()
    mask = (preds["store_nbr"] == store) & (preds["family"] == family)
    df = preds[mask].sort_values("date").copy()

    if df.empty:
        return pd.DataFrame(columns=["date", "yhat", "ci_lower", "ci_upper"])

    yhat = df["ensemble_pred"].values
    actual = df["sales"].values
    mae = float(np.mean(np.abs(actual - yhat)))
    ci = 1.5 * mae

    result = pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d").values,
        "yhat": np.round(yhat, 2),
        "ci_lower": np.round(np.maximum(yhat - ci, 0), 2),
        "ci_upper": np.round(yhat + ci, 2),
    })
    return result


def get_mape(store: int, family: str) -> float:
    preds = load_val_preds()
    mask = (preds["store_nbr"] == store) & (preds["family"] == family)
    df = preds[mask]
    if df.empty or df["sales"].sum() == 0:
        return 0.0
    yhat = df["ensemble_pred"].values
    actual = df["sales"].values
    nonzero = actual > 0
    if not nonzero.any():
        return 0.0
    mape = float(np.mean(np.abs((actual[nonzero] - yhat[nonzero]) / actual[nonzero])) * 100)
    return round(mape, 1)


def lgbm_predict_what_if(
    store: int,
    family: str,
    onpromotion_override: float | None = None,
    oil_override: float | None = None,
    holiday_override: bool | None = None,
) -> pd.DataFrame:
    """Run LGBM on val features with overridden inputs for what-if."""
    val_feats = load_val_features()
    mask = (val_feats["store_nbr"] == store) & (val_feats["family"] == family)
    df = val_feats[mask].copy().sort_values("date")

    if df.empty:
        return pd.DataFrame(columns=["date", "sales"])

    if onpromotion_override is not None:
        # onpromotion_override is a multiplier (e.g. 1.5 = 50% more promo items)
        # floor at 10 so zero-promo items still see a meaningful boost
        df["onpromotion"] = (df["onpromotion"].clip(lower=10) * onpromotion_override).round().astype(int)
    if oil_override is not None:
        df["oil_price"] = float(oil_override)
    if holiday_override is not None:
        df["is_national_holiday"] = int(holiday_override)

    model = load_lgbm()
    features = load_lgbm_features()
    X = df[[f for f in features if f in df.columns]]
    log_pred = model.predict(X)
    sales = _log_to_linear(log_pred)

    return pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d").values,
        "sales": np.round(sales, 2),
    })
