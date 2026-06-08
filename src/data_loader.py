import pickle
from functools import lru_cache
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent.parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"


@lru_cache(maxsize=1)
def load_train() -> pd.DataFrame:
    return pd.read_parquet(DATA / "train_features.parquet")


@lru_cache(maxsize=1)
def load_val_features() -> pd.DataFrame:
    return pd.read_parquet(DATA / "val_features.parquet")


@lru_cache(maxsize=1)
def load_val_preds() -> pd.DataFrame:
    return pd.read_parquet(MODELS / "ensemble_val_preds.parquet")


@lru_cache(maxsize=1)
def load_inventory_params() -> pd.DataFrame:
    return pd.read_parquet(MODELS / "inventory_params.parquet")


@lru_cache(maxsize=1)
def load_shap_importance() -> pd.DataFrame:
    return pd.read_parquet(MODELS / "shap_importance.parquet")


@lru_cache(maxsize=1)
def load_shap_by_family() -> pd.DataFrame:
    return pd.read_parquet(MODELS / "shap_by_family.parquet")


@lru_cache(maxsize=1)
def load_abc() -> pd.DataFrame:
    return pd.read_parquet(MODELS / "abc_classification.parquet")


@lru_cache(maxsize=1)
def load_current_stock() -> pd.DataFrame:
    return pd.read_parquet(MODELS / "current_stock.parquet")


@lru_cache(maxsize=1)
def _load_lgbm_pkg() -> dict:
    with open(MODELS / "lgbm_prophet.pkl", "rb") as f:
        return pickle.load(f)


def load_lgbm():
    return _load_lgbm_pkg()["model"]


def load_lgbm_features() -> list[str]:
    return _load_lgbm_pkg()["features"]


@lru_cache(maxsize=1)
def load_ensemble_config() -> dict:
    with open(MODELS / "ensemble_config.pkl", "rb") as f:
        return pickle.load(f)


def get_families() -> list[str]:
    return sorted(load_train()["family"].unique().tolist())


def get_stores() -> list[int]:
    return sorted(load_train()["store_nbr"].unique().tolist())


LGBM_FEATURES = [
    "store_nbr", "family_enc", "type_enc", "city_enc", "state_enc", "cluster",
    "onpromotion", "transactions", "oil_price",
    "year", "month", "weekofyear", "dayofweek", "dayofmonth",
    "is_weekend", "is_payday", "is_national_holiday", "is_local_holiday",
    "days_after_earthquake",
    "lag_7", "lag_14", "lag_28", "lag_56",
    "roll_mean_7", "roll_mean_14", "roll_mean_28",
    "roll_std_7", "roll_std_14", "roll_std_28",
    "prophet_trend", "prophet_weekly", "prophet_yearly", "prophet_yhat",
]
