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

# ForecastIQ: Store Sales Forecasting and Inventory Optimization

Demand forecasting and inventory optimization system built for the *Data Mining* university project, using a hybrid ML pipeline (LightGBM + Prophet + LSTM) on the Kaggle [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) dataset.

**Live demo:** [huggingface.co/spaces/ArslanV/predikcijaPotraznje](https://huggingface.co/spaces/ArslanV/predikcijaPotraznje)

---

## Features

| Feature | Description |
|---|---|
| Demand forecast | Daily sales forecast with 95% confidence intervals per store & product family |
| Anomaly detection | Detects unexpected demand spikes and drops (IQR + Isolation Forest) |
| SHAP explanations | Feature-level explanations for each prediction |
| Inventory optimization | EOQ + safety stock calculation with configurable lead time and service level |
| What-If simulator | Scenario analysis: change promotions, oil price, or holidays and see the forecast impact |
| Purchase orders | Auto-generated order quantities with stockout cost estimator, exportable as CSV, Excel, or PDF |
| ABC-XYZ matrix | Inventory risk segmentation by value tier (ABC) and demand variability (XYZ) |
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

### Model performance (validation: Aug 1-15, 2017)

Metric: RMSLE (Root Mean Squared Log Error), lower is better.

| Model | RMSLE |
|---|---|
| Naive baseline | 0.5690 |
| LSTM | 0.4140 |
| LightGBM + Prophet features | 0.3695 |
| **Ensemble (α=0.90)** | **0.3689** |

The ensemble improves 35.2% over the naive baseline.

---

## Dataset

Store Sales - Time Series Forecasting ([Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting))

- 54 stores across Ecuador
- 33 product families
- Daily sales from 2013 to 2017 (3M+ rows)
- External factors: oil prices, national/local holidays, promotions, transactions

---

## Project Structure

```
├── app.py                  Landing page with live KPIs
├── pages/
│   ├── 1_📊_Forecast.py    Forecast + anomaly detection + SHAP + MAPE by family
│   ├── 2_🎛️_What_If.py    Scenario simulator with multi-scenario comparison
│   ├── 3_📋_Orders.py      Purchase order generator with stockout cost estimator
│   ├── 4_🔲_ABC_XYZ.py    ABC-XYZ inventory risk matrix
│   └── 5_📤_Upload.py      CSV upload & column mapping
├── src/
│   ├── data_loader.py      Cached data loading functions
│   ├── model_inference.py  Forecast, MAPE, What-If inference
│   ├── anomaly.py          IQR + Isolation Forest anomaly detection
│   ├── inventory.py        EOQ, safety stock, order generation
│   ├── po_generator.py     Excel purchase order export
│   ├── pdf_generator.py    PDF purchase order export
│   └── ui.py               Shared sidebar branding and chart styles
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

> **Note:** Model files and processed data are stored via Git LFS (~232 MB total).
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
- `fpdf2 >= 2.7` (PDF export)
