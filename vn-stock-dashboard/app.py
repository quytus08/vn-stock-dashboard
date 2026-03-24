import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(__file__))
from data.fetch import fetch_stock_data, get_popular_tickers
from models.predictor import predict_trend

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VN Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0d1117; }
    .block-container { padding: 1.5rem 2rem; }

    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8b949e;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .up   { color: #3fb950; }
    .down { color: #f85149; }
    .neu  { color: #d29922; }

    .predict-box {
        background: #161b22;
        border-left: 4px solid #388bfd;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-top: 1rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
    }
    .stSelectbox > div { border-color: #30363d !important; }
    h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 VN Stock Dashboard")
    st.markdown("---")

    popular = get_popular_tickers()
    ticker = st.selectbox("🔎 Mã cổ phiếu", popular, index=0)

    custom = st.text_input("Hoặc nhập mã khác (VD: FPT, MWG...)", "").upper().strip()
    if custom:
        ticker = custom

    st.markdown("---")
    period_map = {
        "1 tháng": 30,
        "3 tháng": 90,
        "6 tháng": 180,
        "1 năm":   365,
        "2 năm":   730,
    }
    period_label = st.selectbox("📅 Khoảng thời gian", list(period_map.keys()), index=2)
    days = period_map[period_label]

    st.markdown("---")
    show_ma   = st.checkbox("📉 Hiển thị MA20 / MA50", value=True)
    show_pred = st.checkbox("🤖 Hiển thị dự đoán ML", value=True)
    predict_days = st.slider("Dự đoán (ngày tới)", 5, 30, 10) if show_pred else 10

    st.markdown("---")
    st.caption("Data: vnstock (VNDIRECT/SSI)\nPortfolio project — Data Analytics")

# ── Fetch data ──────────────────────────────────────────────────────────────────
end_date   = datetime.today()
start_date = end_date - timedelta(days=days)

with st.spinner(f"Đang tải dữ liệu {ticker}..."):
    df = fetch_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if df is None or df.empty:
    st.error(f"❌ Không tìm thấy dữ liệu cho mã **{ticker}**. Kiểm tra lại tên mã cổ phiếu.")
    st.stop()

# ── Derived metrics ─────────────────────────────────────────────────────────────
latest       = df.iloc[-1]
prev         = df.iloc[-2] if len(df) > 1 else latest
price        = latest["close"]
change       = price - prev["close"]
change_pct   = (change / prev["close"]) * 100
vol          = latest["volume"]
high_52w     = df["high"].max()
low_52w      = df["low"].min()

color_cls    = "up" if change >= 0 else "down"
arrow        = "▲" if change >= 0 else "▼"

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown(f"## {ticker} &nbsp; <span class='{color_cls}'>{arrow} {change_pct:+.2f}%</span>", unsafe_allow_html=True)
st.caption(f"Cập nhật đến: {latest.name.date() if hasattr(latest.name, 'date') else latest.name}")

# ── Metrics row ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
def metric_card(col, label, value, color=""):
    col.markdown(f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color}">{value}</div>
    </div>""", unsafe_allow_html=True)

metric_card(c1, "Giá hiện tại", f"{price:,.0f}", color_cls)
metric_card(c2, "Thay đổi",     f"{change:+,.0f}", color_cls)
metric_card(c3, "Khối lượng",   f"{vol/1e6:.2f}M")
metric_card(c4, "Đỉnh 52 tuần", f"{high_52w:,.0f}", "up")
metric_card(c5, "Đáy 52 tuần",  f"{low_52w:,.0f}",  "down")

st.markdown("<br>", unsafe_allow_html=True)

# ── Moving averages ─────────────────────────────────────────────────────────────
if show_ma:
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()

# ── ML Prediction ───────────────────────────────────────────────────────────────
pred_df = None
if show_pred:
    pred_df = predict_trend(df, predict_days)

# ── Candlestick Chart ───────────────────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.72, 0.28],
    vertical_spacing=0.04,
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index, open=df["open"], high=df["high"],
    low=df["low"], close=df["close"],
    name=ticker,
    increasing_line_color="#3fb950",
    decreasing_line_color="#f85149",
    increasing_fillcolor="#3fb950",
    decreasing_fillcolor="#f85149",
), row=1, col=1)

# MA lines
if show_ma:
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20",
        line=dict(color="#d29922", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50",
        line=dict(color="#388bfd", width=1.2, dash="dash")), row=1, col=1)

# ML Prediction line
if pred_df is not None and not pred_df.empty:
    fig.add_trace(go.Scatter(
        x=pred_df.index, y=pred_df["predicted"],
        name="Dự đoán ML",
        line=dict(color="#bc8cff", width=2, dash="dot"),
        mode="lines+markers",
        marker=dict(size=4, color="#bc8cff"),
    ), row=1, col=1)
    # Shaded area under prediction
    fig.add_trace(go.Scatter(
        x=list(pred_df.index) + list(pred_df.index[::-1]),
        y=list(pred_df["upper"]) + list(pred_df["lower"][::-1]),
        fill="toself", fillcolor="rgba(188,140,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Vùng dự đoán", showlegend=False,
    ), row=1, col=1)

# Volume
colors_vol = ["#3fb950" if c >= o else "#f85149" for c, o in zip(df["close"], df["open"])]
fig.add_trace(go.Bar(
    x=df.index, y=df["volume"],
    name="Volume",
    marker_color=colors_vol,
    marker_line_width=0,
    opacity=0.7,
), row=2, col=1)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono, monospace", color="#c9d1d9", size=11),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10),
    height=560,
)
fig.update_xaxes(gridcolor="#21262d", showgrid=True)
fig.update_yaxes(gridcolor="#21262d", showgrid=True)

st.plotly_chart(fig, use_container_width=True)

# ── Prediction summary ──────────────────────────────────────────────────────────
if show_pred and pred_df is not None:
    last_pred = pred_df["predicted"].iloc[-1]
    pred_change = ((last_pred - price) / price) * 100
    trend_word = "TĂNG 📈" if pred_change > 0 else "GIẢM 📉"
    st.markdown(f"""<div class="predict-box">
        🤖 <b>ML Prediction ({predict_days} ngày)</b>&nbsp;&nbsp;|&nbsp;&nbsp;
        Dự đoán giá: <b>{last_pred:,.0f}</b> &nbsp;({pred_change:+.1f}%) &nbsp;→ Xu hướng: <b>{trend_word}</b>
        <br><span style="color:#8b949e;font-size:0.78rem;">⚠️ Mô hình Linear Regression đơn giản — chỉ mang tính tham khảo, không phải tư vấn tài chính.</span>
    </div>""", unsafe_allow_html=True)

# ── Raw data table ──────────────────────────────────────────────────────────────
with st.expander("📋 Xem dữ liệu thô"):
    st.dataframe(
        df[["open","high","low","close","volume"]].tail(30).sort_index(ascending=False),
        use_container_width=True,
    )
    csv = df.to_csv().encode("utf-8")
    st.download_button("⬇️ Tải CSV", csv, f"{ticker}_data.csv", "text/csv")
