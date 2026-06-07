import streamlit as st

HOVERLABEL = dict(bgcolor="#1e293b", font_color="#e2e8f0", bordercolor="#334155")


def render_sidebar():
    st.sidebar.markdown("""
<div style="
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%);
    border-radius: 10px;
    padding: 16px 18px 12px 18px;
    margin-bottom: 8px;
">
    <div style="font-size:1.35rem; font-weight:800; color:#e2e8f0; letter-spacing:0.5px;">
        ◈ ForecastIQ
    </div>
    <div style="font-size:0.75rem; color:#64748b; margin-top:3px; letter-spacing:0.3px;">
        Demand &nbsp;·&nbsp; Inventory &nbsp;·&nbsp; Orders
    </div>
</div>
""", unsafe_allow_html=True)
