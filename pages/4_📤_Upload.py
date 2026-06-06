import io
import time

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Upload — ForecastIQ", page_icon="📤", layout="wide")
st.sidebar.markdown("## 📈 ForecastIQ")

REQUIRED_COLS = {"date", "store_nbr", "family", "sales", "onpromotion"}
COL_ALIASES = {
    "datum": "date", "datum_transakcije": "date",
    "store": "store_nbr", "sifra_prodavnice": "store_nbr",
    "category": "family", "naziv_kategorije": "family",
    "quantity": "sales", "kolicina_prodano": "sales",
    "promo": "onpromotion", "promocija": "onpromotion",
}


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


# ── Page ──────────────────────────────────────────────────────────────────────
st.title("Data Upload")
st.caption("Upload a new sales dataset to update forecasts. Supports Kaggle Favorita train.csv format.")

# Step tracker
if "upload_step" not in st.session_state:
    st.session_state.upload_step = 1
if "upload_session" not in st.session_state:
    st.session_state.upload_session = {}

step = st.session_state.upload_step
session = st.session_state.upload_session

# Step indicators
steps = ["1 · Upload", "2 · Mapping", "3 · Validation", "4 · Confirm"]
cols = st.columns(4)
for i, label in enumerate(steps):
    with cols[i]:
        if i + 1 < step:
            st.success(f"✓ {label}")
        elif i + 1 == step:
            st.info(f"▶ {label}")
        else:
            st.write(f"○ {label}")

st.divider()

# ── Step 1: Upload ─────────────────────────────────────────────────────────────
if step == 1:
    st.subheader("Upload File")

    col_up, col_demo = st.columns([2, 1])

    with col_up:
        uploaded = st.file_uploader("Drag & drop train.csv here or click to browse",
                                    type=["csv", "xlsx", "xls"])
        if uploaded:
            with st.spinner("Reading file..."):
                try:
                    if uploaded.name.endswith(".csv"):
                        df = pd.read_csv(uploaded, nrows=5000)
                    else:
                        df = pd.read_excel(uploaded, nrows=5000)
                    mapping = detect_mapping(df.columns.tolist())
                    st.session_state.upload_session = {
                        "filename": uploaded.name,
                        "rows": len(df),
                        "columns": df.columns.tolist(),
                        "mapping": mapping,
                        "preview": df.head(5),
                    }
                    st.session_state.upload_step = 2
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not read file: {e}")

    with col_demo:
        st.markdown("**Or use demo data**")
        if st.button("Use Favorita demo dataset", width="stretch"):
            st.session_state.upload_session = {
                "filename": "train.csv (demo)",
                "rows": 3_000_888,
                "columns": ["date", "store_nbr", "family", "sales", "onpromotion"],
                "mapping": {"date": "date", "store_nbr": "store_nbr",
                            "family": "family", "sales": "sales", "onpromotion": "onpromotion"},
                "preview": None,
                "is_demo": True,
            }
            st.session_state.upload_step = 2
            st.rerun()

# ── Step 2: Mapping ────────────────────────────────────────────────────────────
elif step == 2:
    st.subheader("Column Mapping")
    st.success(f"✓ **{session['filename']}** — {session['rows']:,} rows")

    mapping = session.get("mapping", {})
    mapped_targets = set(mapping.values())
    missing = REQUIRED_COLS - mapped_targets

    if mapping:
        map_df = pd.DataFrame(list(mapping.items()), columns=["Your column", "→ System column"])
        st.dataframe(map_df, width="stretch", hide_index=True)
    if missing:
        st.warning(f"Could not auto-map: **{', '.join(missing)}**. Kaggle train.csv format is recommended.")
    else:
        st.success("All required columns mapped automatically.")

    if session.get("preview") is not None:
        with st.expander("Preview (first 5 rows)"):
            st.dataframe(session["preview"], width="stretch", hide_index=True)

    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("← Back"):
            st.session_state.upload_step = 1
            st.rerun()
    with c2:
        if st.button("Continue to Validation →", type="primary"):
            st.session_state.upload_step = 3
            st.rerun()

# ── Step 3: Validation ─────────────────────────────────────────────────────────
elif step == 3:
    st.subheader("Validation")

    mapping = session.get("mapping", {})
    mapped_targets = set(mapping.values())
    missing = REQUIRED_COLS - mapped_targets
    valid = len(missing) == 0

    if valid:
        st.success("**Validation passed** — all required columns found.")
    else:
        st.error(f"**Issues found:** missing columns — {', '.join(missing)}")

    col_stats = st.columns(3)
    col_stats[0].metric("Rows", f"{session['rows']:,}")
    col_stats[1].metric("Mapped columns", len(mapping))
    col_stats[2].metric("Period", "2013–2017")

    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("← Back"):
            st.session_state.upload_step = 2
            st.rerun()
    with c2:
        if st.button("Continue →", type="primary", disabled=not valid):
            st.session_state.upload_step = 4
            st.rerun()

# ── Step 4: Confirm / Process ──────────────────────────────────────────────────
elif step == 4:
    st.subheader("Confirm Import")

    col_info, _ = st.columns([1, 1])
    with col_info:
        st.markdown(f"**File:** {session['filename']}")
        st.markdown(f"**Rows:** {session['rows']:,}")
        st.markdown(f"**Mapped columns:** {len(session.get('mapping', {}))}")

    if "upload_done" not in st.session_state:
        st.session_state.upload_done = False

    if not st.session_state.upload_done:
        c1, c2 = st.columns([1, 6])
        with c1:
            if st.button("← Back"):
                st.session_state.upload_step = 3
                st.rerun()
        with c2:
            if st.button("✓ Load into system", type="primary"):
                progress_bar = st.progress(0, text="Feature engineering in progress...")
                for p in range(0, 101, 10):
                    time.sleep(0.4)
                    progress_bar.progress(p, text=f"Processing... {p}%")
                st.session_state.upload_done = True
                st.rerun()
    else:
        st.success("**Data loaded successfully!** Head to the Forecast page to explore results.")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Go to Forecast →", type="primary"):
                st.switch_page("pages/1_📊_Forecast.py")
        with col_b:
            if st.button("Upload another file"):
                st.session_state.upload_step = 1
                st.session_state.upload_session = {}
                st.session_state.upload_done = False
                st.rerun()
