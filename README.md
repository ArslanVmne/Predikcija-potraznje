---
title: ForecastIQ
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.45.0"
app_file: app.py
pinned: false
---

# ForecastIQ — Store Sales Forecasting & Inventory Optimization

Intelligent demand forecasting and inventory optimization system built for the *Demand Mining* university project.
Uses a hybrid ML pipeline (LightGBM + Prophet + LSTM) on the Kaggle [Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) dataset.

**Live demo:** [huggingface.co/spaces/ArslanV/predikcijaPotraznje](https://huggingface.co/spaces/ArslanV/predikcijaPotraznje)

---

## Features

| Feature | Description |
|---|---|
| Demand forecast | Daily sales forecast with 95% confidence intervals per store & product family |
| Anomaly detection | Detects unexpected demand spikes and drops (IQR + Isolation Forest) |
| SHAP explanations | Feature-level explanations for each prediction |
| Inventory optimization | EOQ + safety stock calculation with configurable lead time and service level |
| What-If simulator | Scenario analysis — change promotions, oil price, or holidays and see the forecast impact |
| Purchase orders | Auto-generated order quantities, exportable as CSV or Excel |
| Data upload | Upload custom sales CSV and map columns to the system schema |

---

## ML Pipeline

```
Raw data
  └── 01_eda.ipynb               Exploratory data analysis
  └── 02_feature_engineering.ipynb  Lag features, rolling stats, Prophet components, encodings
  └── 03_modeling.ipynb          LightGBM baseline + LightGBM with Prophet features
  └── 04_prophet.ipynb           Facebook Prophet model
  └── 05_lstm.ipynb              LSTM sequence model (PyTorch)
  └── 06_ensemble.ipynb          Weighted ensemble (LightGBM + LSTM)
  └── 07_shap.ipynb              SHAP explainability
  └── 08_inventory.ipynb         EOQ, safety stock, ABC classification
```

### Model performance (validation: Aug 1–15, 2017)

| Model | MAPE |
|---|---|
| LightGBM baseline | ~18% |
| LightGBM + Prophet features | ~14% |
| LSTM | ~16% |
| **Ensemble** | **~12%** |

---

## Dataset

**Store Sales — Time Series Forecasting** ([Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting))

- 54 stores across Ecuador
- 33 product families
- Daily sales from 2013 to 2017 (3M+ rows)
- External factors: oil prices, national/local holidays, promotions, transactions

---

## Project Structure

```
├── app.py                  About / landing page
├── pages/
│   ├── 1_📊_Forecast.py    Forecast + anomaly detection + SHAP
│   ├── 2_🎛️_What_If.py    Scenario simulator
│   ├── 3_📋_Orders.py      Purchase order generator
│   └── 4_📤_Upload.py      CSV upload & column mapping
├── src/
│   ├── data_loader.py      Cached data loading functions
│   ├── model_inference.py  Forecast, MAPE, What-If inference
│   ├── anomaly.py          IQR + Isolation Forest anomaly detection
│   └── inventory.py        EOQ, safety stock, order generation
├── models/                 Pre-trained model artifacts (LFS)
├── data/processed/         Processed parquet files (LFS)
├── notebooks/              Full ML pipeline (01–08)
└── requirements.txt
```

---

## Run Locally

```bash
git clone https://github.com/ArslanVmne/Predikcija-potraznje.git
cd Predikcija-potraznje
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** `data/processed/train_features.parquet` is stored via Git LFS (173 MB).
> Make sure you have [Git LFS](https://git-lfs.com) installed and run `git lfs pull` after cloning.

---

## Requirements

See [requirements.txt](requirements.txt). Key dependencies:

- `streamlit >= 1.35`
- `lightgbm >= 4.0`
- `prophet >= 1.1`
- `shap >= 0.44`
- `scikit-learn >= 1.3`
- `plotly >= 5.18`
- `pandas >= 2.0`
