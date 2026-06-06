import math

import numpy as np
import pandas as pd

from src.data_loader import load_inventory_params, load_val_preds


def compute_order_quantity(
    mean_daily: float,
    std_daily: float,
    lead_time: int = 7,
    service_level: float = 0.95,
    current_stock: float = 0.0,
    min_order: int = 50,
    ml_forecast: float | None = None,
) -> dict:
    z = {0.90: 1.28, 0.95: 1.645, 0.99: 2.326}.get(service_level, 1.645)
    safety_stock = z * std_daily * math.sqrt(lead_time)
    rop = mean_daily * lead_time + safety_stock
    # Use ML ensemble forecast if available, otherwise fall back to historical mean
    forecast_demand = ml_forecast if ml_forecast is not None else mean_daily * lead_time
    order_qty = max(int(round(forecast_demand + safety_stock - current_stock)), min_order)
    return {
        "safety_stock": round(safety_stock, 1),
        "rop": round(rop, 1),
        "order_qty": order_qty,
        "forecast_demand": round(forecast_demand, 1),
    }


def build_orders(
    lead_time: int = 7,
    service_level: float = 0.95,
    min_order: int = 50,
    store_filter: int | None = None,
) -> list[dict]:
    params = load_inventory_params()
    val = load_val_preds()

    # aggregate forecast by store/family over val period
    # ensemble_pred is already in linear space
    agg = (
        val.groupby(["store_nbr", "family"])
        .agg(forecast=("ensemble_pred", "sum"))
        .reset_index()
    )

    if store_filter is not None:
        params = params[params["store_nbr"] == store_filter]
        agg = agg[agg["store_nbr"] == store_filter]

    merged = params.merge(agg, on=["store_nbr", "family"], how="left")
    merged["forecast"] = merged["forecast"].fillna(merged["mean_daily"] * lead_time)

    val_period_days = int(val["date"].nunique()) if "date" in val.columns else lead_time

    orders = []
    for _, row in merged.iterrows():
        # Scale ensemble forecast from val period to lead time window
        ml_forecast = float(row["forecast"]) / max(val_period_days, 1) * lead_time
        info = compute_order_quantity(
            mean_daily=row["mean_daily"],
            std_daily=row["std_daily"],
            lead_time=lead_time,
            service_level=service_level,
            current_stock=row["safety_stock"],
            min_order=min_order,
            ml_forecast=ml_forecast,
        )
        abc = row["ABC"]
        if abc == "A":
            status, status_class = "Urgent", "bgr"
        elif abc == "B":
            status, status_class = "Regular", "bgb"
        else:
            status, status_class = "Seasonal", "bgb"

        # Demo unit price: cost-per-unit scaled to realistic retail range
        raw = row["annual_cost"] / max(row["annual_demand"], 1)
        unit_price = round(max(raw * 500, 0.50), 2)
        orders.append({
            "id": f"{int(row['store_nbr'])}-{row['family']}",
            "product": row["family"],
            "store": f"Store {int(row['store_nbr'])}",
            "store_nbr": int(row["store_nbr"]),
            "ml_forecast": round(info["forecast_demand"], 0),
            "eoq": round(float(row["EOQ"]), 0),
            "safety_stock": round(info["safety_stock"], 0),
            "suggested": info["order_qty"],
            "qty": info["order_qty"],
            "unit_price": round(unit_price, 2),
            "total": round(info["order_qty"] * unit_price, 2),
            "status": status,
            "status_class": status_class,
            "abc": abc,
        })

    orders.sort(key=lambda x: ({"bgr": 0, "bgy": 1, "bgb": 2}.get(x["status_class"], 3)))
    return orders
