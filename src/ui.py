import streamlit as st

HOVERLABEL = dict(bgcolor="#1e293b", font_color="#e2e8f0", bordercolor="#334155")

STATUS_EMOJI = {
    "Critical":  "🔴 Critical",
    "Order Now": "🟡 Order Now",
    "Monitor":   "🔵 Monitor",
    "OK":        "🟢 OK",
}


def strip_status_emoji(s: str) -> str:
    """Remove leading emoji + space from a status string for plain-text export."""
    return s.split(" ", 1)[-1] if s and s[0] in "🔴🟡🔵🟢" else s


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
