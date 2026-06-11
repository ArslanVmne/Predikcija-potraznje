import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_val_preds
from src.model_inference import get_forecast, lgbm_predict_what_if
from src.ui import HOVERLABEL, render_sidebar

st.set_page_config(page_title="Upload | ForecastIQ", page_icon="📤", layout="wide")
render_sidebar()

REQUIRED_COLS = {"date", "store_nbr", "family", "sales", "onpromotion"}
COL_ALIASES = {
    "datum": "date", "datum_transakcije": "date",
    "store": "store_nbr", "sifra_prodavnice": "store_nbr",
    "category": "family", "naziv_kategorije": "family",
    "quantity": "sales", "kolicina_prodano": "sales",
    "promo": "onpromotion", "promocija": "onpromotion",
}

CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", size=12),
    margin=dict(t=20, r=20, b=40, l=60),
    hoverlabel=HOVERLABEL,
)


def detect_mapping(columns: list[str]) -> dict[str, str]:
    mapping = {}
    cols_lower = {c.lower(): c for c in columns}
    for req in REQUIRED_COLS:
        if req in cols_lower:
            mapping[cols_lower[req]] = req
        else:
            for alias, target in COL_ALIASES.items():
                if target == req and alias in cols_lower:
                    mapping[cols_lower[alias]] = req
                    break
    return mapping


def read_uploaded(file) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        return pd.read_csv(file, nrows=50_000)
    return pd.read_excel(file, nrows=50_000)


# ── Session state init ────────────────────────────────────────────────────────
if "upload_step" not in st.session_state:
    st.session_state.upload_step = 1
if "upload_df" not in st.session_state:
    st.session_state.upload_df = None
if "upload_meta" not in st.session_state:
    st.session_state.upload_meta = {}

step = st.session_state.upload_step

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Data Upload")
st.caption("Upload a sales dataset to explore its structure and validate it for forecasting.")

# Step indicators
step_labels = ["1 · Upload", "2 · Mapping", "3 · Explore", "4 · Done"]
cols = st.columns(4)
for i, label in enumerate(step_labels):
    with cols[i]:
        if i + 1 < step:
            st.success(f"✓ {label}")
        elif i + 1 == step:
            st.info(f"▶ {label}")
        else:
            st.write(f"○ {label}")

st.divider()

# ── Step 1 — Upload ───────────────────────────────────────────────────────────
if step == 1:
    st.subheader("Upload a file or download sample data")

    col_up, col_demo = st.columns([3, 2])

    with col_up:
        uploaded = st.file_uploader(
            "Drag & drop your CSV here",
            type=["csv", "xlsx", "xls"],
            help="Expected columns: date, store_nbr, family, sales, onpromotion",
        )
        if uploaded:
            with st.spinner("Reading file..."):
                try:
                    df = read_uploaded(uploaded)
                    mapping = detect_mapping(df.columns.tolist())
                    st.session_state.upload_df = df
                    st.session_state.upload_meta = {
                        "filename": uploaded.name,
                        "mapping": mapping,
                    }
                    st.session_state.upload_step = 2
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not read file: {e}")

    with col_demo:
        st.markdown("**Don't have a file? Download sample data:**")

        with open("data/demo_sales.csv", "rb") as f:
            st.download_button(
                "⬇ demo_sales.csv",
                data=f.read(),
                file_name="demo_sales.csv",
                mime="text/csv",
                use_container_width=True,
                help="Jan-Jul 2017 · Stores 1, 3, 5 · 33 families · 20,988 rows",
            )

        st.caption("Download → inspect → re-upload to see the full pipeline.")

# ── Step 2 — Column Mapping ───────────────────────────────────────────────────
elif step == 2:
    meta = st.session_state.upload_meta
    df = st.session_state.upload_df
    mapping = meta.get("mapping", {})
    mapped_targets = set(mapping.values())
    missing = REQUIRED_COLS - mapped_targets

    st.subheader("Column Mapping")
    st.success(f"✓ **{meta['filename']}**: {len(df):,} rows, {len(df.columns)} columns")

    # Detect if user uploaded the external factors file by mistake
    cols_lower = {c.lower() for c in df.columns}
    is_external_factors = "oil_price" in cols_lower or (
        len(missing) >= 4 and not {"sales", "family"} & cols_lower
    )

    if is_external_factors:
        st.warning(
            "**This looks like an external factors file** (oil prices, holidays, transactions).  \n"
            "The forecasting pipeline requires a **sales file** with columns:  \n"
            "`date · store_nbr · family · sales · onpromotion`  \n\n"
            "Please upload **demo_sales.csv** instead to see the full pipeline."
        )
        st.markdown("**File preview:**")
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)
        if st.button("← Start over", type="primary"):
            st.session_state.upload_step = 1
            st.session_state.upload_df = None
            st.session_state.upload_meta = {}
            st.rerun()
    else:
        col_map, col_prev = st.columns([1, 1])

        with col_map:
            if mapping:
                map_df = pd.DataFrame(
                    [(k, "→", v) for k, v in mapping.items()],
                    columns=["Your column", "", "System column"],
                )
                st.dataframe(map_df, use_container_width=True, hide_index=True)

            if missing:
                st.warning(f"Could not auto-map: **{', '.join(missing)}**")
                for col in missing:
                    choice = st.selectbox(
                        f"Map '{col}' to:",
                        ["(skip)"] + df.columns.tolist(),
                        key=f"manual_{col}",
                    )
                    if choice != "(skip)":
                        mapping[choice] = col
            else:
                st.success("All required columns mapped automatically.")

        with col_prev:
            st.markdown("**Preview (5 rows)**")
            st.dataframe(df.head(5), use_container_width=True, hide_index=True)

        st.session_state.upload_meta["mapping"] = mapping
        mapped_targets = set(mapping.values())
        missing = REQUIRED_COLS - mapped_targets

        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("← Back"):
                st.session_state.upload_step = 1
                st.rerun()
        with c2:
            if st.button("Continue →", type="primary", disabled=bool(missing)):
                st.session_state.upload_step = 3
                st.rerun()

# ── Step 3 — Explore ──────────────────────────────────────────────────────────
elif step == 3:
    meta = st.session_state.upload_meta
    raw_df = st.session_state.upload_df
    mapping = meta["mapping"]

    # Apply column mapping and parse
    df = raw_df.rename(columns=mapping).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)
    df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce")
    df = df.dropna(subset=["date", "store_nbr", "family"])

    date_min = df["date"].min().strftime("%b %d, %Y")
    date_max = df["date"].max().strftime("%b %d, %Y")
    n_stores = df["store_nbr"].nunique()
    n_families = df["family"].nunique()
    n_rows = len(df)
    total_sales = df["sales"].sum()
    missing_pct = round(raw_df.isnull().mean().mean() * 100, 1)

    st.subheader("Data Explorer")

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Rows", f"{n_rows:,}")
    k2.metric("Stores", n_stores)
    k3.metric("Families", n_families)
    k4.metric("Date range", f"{date_min} - {date_max}")
    k5.metric("Missing values", f"{missing_pct}%",
              delta="Clean" if missing_pct < 2 else "Check data",
              delta_color="off" if missing_pct < 2 else "inverse")

    st.divider()

    col_left, col_right = st.columns(2)

    # Sales trend
    with col_left:
        st.markdown("**Daily sales trend**")
        daily = df.groupby("date")["sales"].sum().reset_index()
        fig = go.Figure(go.Scatter(
            x=daily["date"], y=daily["sales"],
            mode="lines", line=dict(color="#2563eb", width=2),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.1)",
        ))
        fig.update_layout(height=220, **CHART_LAYOUT,
                          xaxis=dict(showgrid=False),
                          yaxis=dict(gridcolor="#334155"))
        st.plotly_chart(fig, use_container_width=True)

    # Top families
    with col_right:
        st.markdown("**Top families by total sales**")
        top = (df.groupby("family")["sales"].sum()
                 .sort_values(ascending=True).tail(10).reset_index())
        fig2 = go.Figure(go.Bar(
            x=top["sales"], y=top["family"],
            orientation="h", marker_color="#2563eb",
        ))
        fig2.update_layout(height=220, **CHART_LAYOUT,
                           xaxis=dict(title="Total units", gridcolor="#334155"),
                           yaxis=dict(showgrid=False))
        st.plotly_chart(fig2, use_container_width=True)

    # Store breakdown
    st.markdown("**Sales by store**")
    store_df = df.groupby("store_nbr")["sales"].sum().reset_index().sort_values("sales", ascending=False)
    fig3 = px.bar(store_df, x="store_nbr", y="sales",
                  labels={"store_nbr": "Store", "sales": "Total sales"},
                  color_discrete_sequence=["#2563eb"])
    fig3.update_layout(height=200, **CHART_LAYOUT,
                       xaxis=dict(showgrid=False, tickmode="linear"),
                       yaxis=dict(gridcolor="#334155"))
    st.plotly_chart(fig3, use_container_width=True)

    # Validation summary
    issues = []
    if missing_pct > 5:
        issues.append(f"High missing value rate: {missing_pct}%")
    if df["sales"].min() < 0:
        issues.append("Negative sales values detected")
    if n_families < 3:
        issues.append("Very few product families; check family column")

    if issues:
        st.warning("**Validation warnings:**\n" + "\n".join(f"- {i}" for i in issues))
    else:
        st.success("**Validation passed**: data looks clean and complete.")

    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("← Back"):
            st.session_state.upload_step = 2
            st.rerun()
    with c2:
        if st.button("✓ Confirm & continue", type="primary"):
            st.session_state.upload_step = 4
            st.rerun()

# ── Step 4 — Forecast results ─────────────────────────────────────────────────
elif step == 4:
    meta = st.session_state.upload_meta
    raw_df = st.session_state.upload_df
    mapping = meta["mapping"]

    df_mapped = raw_df.rename(columns=mapping).copy()
    df_mapped["date"] = pd.to_datetime(df_mapped["date"], errors="coerce")
    df_mapped["store_nbr"] = pd.to_numeric(df_mapped["store_nbr"], errors="coerce")
    df_mapped = df_mapped.dropna(subset=["date", "store_nbr", "family"])

    uploaded_stores = sorted(df_mapped["store_nbr"].dropna().astype(int).unique().tolist())
    uploaded_families = sorted(df_mapped["family"].unique().tolist())
    date_max = df_mapped["date"].max()

    # Check which store/family combos exist in the model's val predictions
    val = load_val_preds()
    supported = val[
        val["store_nbr"].isin(uploaded_stores)
    ][["store_nbr", "family"]].drop_duplicates()
    n_supported = len(supported)

    st.success(f"**{meta['filename']}**: {len(df_mapped):,} rows loaded.")
    st.info(
        f"Your data covers **{date_max.strftime('%B %d, %Y')}**. "
        f"The pre-trained ensemble model will forecast the **next 15 days** "
        f"(Aug 1-15, 2017) for **{n_supported} store/family combinations** "
        f"found in your upload."
    )

    if n_supported == 0:
        st.warning("None of the stores in your file are in the model's training set. "
                   "Try uploading demo_sales.csv to see the full pipeline.")
    else:
        # Run inference for all supported combos
        with st.spinner(f"Forecasting {n_supported} store/product combinations..."):
            rows = []
            for _, r in supported.iterrows():
                fc = get_forecast(int(r["store_nbr"]), r["family"])
                if not fc.empty:
                    total = float(fc["yhat"].sum())
                    rows.append({
                        "store_nbr": int(r["store_nbr"]),
                        "family": r["family"],
                        "forecast_15d": round(total, 0),
                    })
            fc_df = pd.DataFrame(rows)

        st.divider()
        st.subheader("Forecast Results: Aug 1-15, 2017")

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Store/product combinations", len(fc_df))
        k2.metric("Total forecasted demand", f"{int(fc_df['forecast_15d'].sum()):,} units")
        k3.metric("Avg per combination", f"{int(fc_df['forecast_15d'].mean()):,} units")

        col_left, col_right = st.columns(2)

        # Top families by forecast
        with col_left:
            st.markdown("**Top families by 15-day forecast**")
            top_fam = (fc_df.groupby("family")["forecast_15d"]
                       .sum().sort_values(ascending=True).tail(10).reset_index())
            fig = go.Figure(go.Bar(
                x=top_fam["forecast_15d"], y=top_fam["family"],
                orientation="h",
                marker_color="#2563eb",
                text=top_fam["forecast_15d"].astype(int),
                textposition="outside",
            ))
            fig.update_layout(height=300, **CHART_LAYOUT,
                              xaxis=dict(title="Forecasted units", gridcolor="#334155"),
                              yaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True)

        # Forecast by store
        with col_right:
            st.markdown("**Forecasted demand by store**")
            by_store = fc_df.groupby("store_nbr")["forecast_15d"].sum().reset_index()
            by_store["store_nbr"] = by_store["store_nbr"].astype(str)
            fig2 = px.bar(by_store, x="store_nbr", y="forecast_15d",
                          labels={"store_nbr": "Store", "forecast_15d": "Forecasted units"},
                          color_discrete_sequence=["#16a34a"])
            fig2.update_layout(height=300, **CHART_LAYOUT,
                               xaxis=dict(showgrid=False),
                               yaxis=dict(gridcolor="#334155"))
            st.plotly_chart(fig2, use_container_width=True)

        # Store uploaded stores in session so Forecast page can pre-filter
        st.session_state["uploaded_stores"] = uploaded_stores

        st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Explore Forecast →", type="primary", use_container_width=True):
            st.switch_page("pages/1_📊_Forecast.py")
    with c2:
        if st.button("View Orders →", use_container_width=True):
            st.switch_page("pages/3_📋_Orders.py")
    with c3:
        if st.button("Upload another file", use_container_width=True):
            st.session_state.upload_step = 1
            st.session_state.upload_df = None
            st.session_state.upload_meta = {}
            st.rerun()
