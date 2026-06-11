import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

st.set_page_config(
    page_title="Tesla Stock Price Predictor",
    layout="wide",
    page_icon="🚗",
    initial_sidebar_state="expanded"
)

DATA_PATHS = [
    os.path.join(os.getcwd(), "TSLA.csv"),
    os.path.join(os.path.expanduser("~"), "Downloads", "TSLA.csv"),
    r"C:\Users\Y.Abihail Nans\Downloads\TSLA.csv"
]
TUNED_MODEL_PATH = os.path.join(os.getcwd(), "best_tuned_model.keras")
SIMPLERNN_MODEL_PATH = os.path.join(os.getcwd(), "best_simplernn_model.h5")
COMPARISON_CSV_PATH = os.path.join(os.getcwd(), "model_comparison_results.csv")

# ── GLOBAL CSS — FULL DARK THEME ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ════════════════════════════════════════════════
   BASE — everything dark, text bright
════════════════════════════════════════════════ */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
.block-container {
    font-family: 'Space Grotesk', sans-serif !important;
    background: #0D0D0D !important;
    color: #F0F0F0 !important;
}
.block-container { padding-top: 24px !important; padding-bottom: 40px !important; }
[data-testid="stMain"] > div { padding-top: 0 !important; }

/* Kill Streamlit chrome */
[data-testid="stHeader"],
[data-testid="stDecoration"],
[data-testid="stToolbar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden !important; height: 0 !important; }

/* Hide default alert/exception boxes — we render our own */
[data-testid="stAlert"],
[data-testid="stException"],
div[class*="stAlert"],
div[class*="Alert"],
.stException { display: none !important; }

/* Hide raw checkboxes (we render styled toggles above them) */
.stCheckbox { display: none !important; }

/* ════════════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #2A2A2A !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 0 !important; }
[data-testid="stSidebar"] * {
    color: #CCCCCC !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider   label {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    color: #888888 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #1E1E1E !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
    color: #F0F0F0 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #888888 !important; }
[data-testid="stSidebar"] [data-baseweb="menu"] { background: #1E1E1E !important; }
[data-testid="stSidebar"] [role="option"] { color: #F0F0F0 !important; }
[data-testid="stSidebar"] input[type="range"] { accent-color: #A5CF83 !important; }
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
    font-size: 10px !important; color: #555 !important;
}

/* ════════════════════════════════════════════════
   METRIC CARDS
════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: #1A1A1A !important;
    border: 1px solid #2E2E2E !important;
    border-radius: 12px !important;
    padding: 20px 22px !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 1.4px !important;
    text-transform: uppercase !important;
    color: #888888 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 30px !important;
    font-weight: 500 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.5px !important;
}
[data-testid="stMetricDelta"] {
    font-size: 13px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ════════════════════════════════════════════════
   HEADINGS
════════════════════════════════════════════════ */
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* ════════════════════════════════════════════════
   DATAFRAME
════════════════════════════════════════════════ */
.stDataFrame,
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #2A2A2A !important;
}
[data-testid="stDataFrame"] th {
    background: #1E1E1E !important;
    color: #AAAAAA !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #333 !important;
}
[data-testid="stDataFrame"] td {
    background: #141414 !important;
    color: #E8E8E8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    border-bottom: 1px solid #1E1E1E !important;
}
[data-testid="stDataFrame"] tr:hover td { background: #1C1C1C !important; }

/* ════════════════════════════════════════════════
   CUSTOM COMPONENTS
════════════════════════════════════════════════ */
.page-header {
    padding: 16px 0 20px;
    border-bottom: 1px solid #2A2A2A;
    margin-bottom: 28px;
}
.panel-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    border-left: 4px solid #A5CF83;
    padding-left: 12px;
    margin: 32px 0 18px 0;
    line-height: 1.3;
}

/* KPI accent top bars */
div[data-testid="column"]:nth-child(1) [data-testid="stMetric"] { border-top: 3px solid #A5CF83 !important; }
div[data-testid="column"]:nth-child(2) [data-testid="stMetric"] { border-top: 3px solid #F0E76F !important; }
div[data-testid="column"]:nth-child(3) [data-testid="stMetric"] { border-top: 3px solid #ECB65F !important; }
div[data-testid="column"]:nth-child(4) [data-testid="stMetric"] { border-top: 3px solid #E89951 !important; }

/* Prediction box */
.pred-box {
    background: #1A1A1A;
    border: 1px solid #2E2E2E;
    border-top: 4px solid #A5CF83;
    border-radius: 14px;
    padding: 28px 24px 24px;
    text-align: center;
    margin-bottom: 18px;
}
.pred-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888888;
    font-family: 'Space Grotesk', sans-serif;
}
.pred-price {
    font-family: 'DM Mono', monospace;
    font-size: 56px;
    font-weight: 500;
    letter-spacing: -2px;
    color: #FFFFFF;
    line-height: 1.05;
    margin: 10px 0 6px;
}
.pred-price em { color: #A5CF83; font-style: normal; }
.pred-change-pos {
    font-family: 'DM Mono', monospace;
    font-size: 16px;
    font-weight: 500;
    color: #A5CF83;
    margin-top: 4px;
}
.pred-change-neg {
    font-family: 'DM Mono', monospace;
    font-size: 16px;
    font-weight: 500;
    color: #E89951;
    margin-top: 4px;
}
.conf-badge {
    display: inline-block;
    background: rgba(165,207,131,0.15);
    border: 1px solid rgba(165,207,131,0.4);
    color: #A5CF83;
    font-size: 13px;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    padding: 6px 20px;
    border-radius: 20px;
    margin-top: 14px;
}

/* MA toggle pills */
.toggle-row {
    display: flex;
    gap: 10px;
    align-items: center;
    margin-bottom: 14px;
    flex-wrap: wrap;
}
.ma-toggle {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    border: 1px solid #333333;
    background: #1A1A1A;
    color: #CCCCCC;
}
.ma-toggle .dot { width: 9px; height: 9px; border-radius: 50%; }
.ma-toggle.active-sage   { background: rgba(165,207,131,0.12); border-color: #A5CF83; color: #A5CF83; }
.ma-toggle.active-amber1 { background: rgba(236,182,95,0.12);  border-color: #ECB65F; color: #ECB65F; }
.ma-toggle.active-amber2 { background: rgba(232,153,81,0.12);  border-color: #E89951; color: #E89951; }

/* Model metric boxes */
.model-metric {
    background: #1A1A1A;
    border: 1px solid #2E2E2E;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.mm-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #888888;
    font-family: 'Space Grotesk', sans-serif;
}
.mm-val {
    font-family: 'DM Mono', monospace;
    font-size: 26px;
    font-weight: 500;
    color: #FFFFFF;
    margin-top: 6px;
}

/* Improvement bar */
.improve-bar {
    background: rgba(165,207,131,0.08);
    border: 1px solid rgba(165,207,131,0.25);
    border-radius: 10px;
    padding: 16px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 16px;
    margin-bottom: 10px;
}
.ib-label {
    font-size: 14px;
    font-weight: 600;
    color: #CCCCCC;
    font-family: 'Space Grotesk', sans-serif;
}
.ib-val {
    font-family: 'DM Mono', monospace;
    font-size: 22px;
    font-weight: 500;
    color: #A5CF83;
}

/* Error / info box */
.warn-box {
    background: #1E1500;
    border: 1px solid #ECB65F;
    border-left: 4px solid #E89951;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    font-weight: 500;
    color: #F5D080;
    font-family: 'Space Grotesk', sans-serif;
    margin-bottom: 12px;
}
.warn-box strong { color: #ECB65F; font-weight: 700; }

.info-box {
    background: #111111;
    border: 1px solid #2E2E2E;
    border-radius: 10px;
    padding: 20px 18px;
    font-size: 14px;
    font-weight: 500;
    color: #888888;
    text-align: center;
    font-family: 'Space Grotesk', sans-serif;
}
.info-box code {
    background: #222222;
    color: #A5CF83;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
}

/* Divider */
.divider { border: none; border-top: 1px solid #1E1E1E; margin: 28px 0; }

/* Footer */
.page-footer {
    text-align: center;
    padding: 24px 0 36px;
    font-size: 12px;
    font-weight: 500;
    color: #555555;
    font-family: 'Space Grotesk', sans-serif;
    border-top: 1px solid #1E1E1E;
    margin-top: 36px;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for path in DATA_PATHS:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
                return df.sort_index()
            except Exception:
                try:
                    df = pd.read_csv(path, parse_dates=True, index_col=0)
                    return df.sort_index()
                except Exception as e:
                    raise IOError(f"Failed to parse TSLA.csv at {path}: {e}")
    raise FileNotFoundError("TSLA.csv not found. Place it in the project folder or Downloads.")

@st.cache_resource
def load_prediction_model(model_path: str):
    # Auto-prefer .keras version over .h5 (no version mismatch issues)
    keras_path = model_path.replace(".h5", ".keras")
    
    # Try .keras first, then original path
    paths_to_try = []
    if os.path.exists(keras_path):
        paths_to_try.append(keras_path)
    if os.path.exists(model_path):
        paths_to_try.append(model_path)
    
    if not paths_to_try:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    last_error = None

    for try_path in paths_to_try:
        # Strategy 1: compile=False (safest for inference)
        try:
            return load_model(try_path, compile=False)
        except Exception as e:
            last_error = e

        # Strategy 2: custom_objects for legacy Keras 2 metric strings
        try:
            custom_objects = {
                "mse": tf.keras.losses.MeanSquaredError(),
                "mean_squared_error": tf.keras.losses.MeanSquaredError(),
                "mae": tf.keras.losses.MeanAbsoluteError(),
                "mean_absolute_error": tf.keras.losses.MeanAbsoluteError(),
            }
            return load_model(try_path, custom_objects=custom_objects, compile=False)
        except Exception as e:
            last_error = e

        # Strategy 3: patch quantization_config directly in .h5 then load
        if try_path.endswith(".h5"):
            try:
                import h5py, json, tempfile, shutil

                tmp_path = try_path.replace(".h5", "_patched.h5")
                shutil.copy2(try_path, tmp_path)

                with h5py.File(tmp_path, 'r+') as f:
                    model_config = json.loads(f.attrs['model_config'])
                    def strip_quant(obj):
                        if isinstance(obj, dict):
                            obj.pop('quantization_config', None)
                            for v in obj.values(): strip_quant(v)
                        elif isinstance(obj, list):
                            for item in obj: strip_quant(item)
                    strip_quant(model_config)
                    f.attrs['model_config'] = json.dumps(model_config)

                model = load_model(tmp_path, compile=False)

                # Auto-save as .keras so next load is instant
                model.save(keras_path)
                return model
            except Exception as e:
                last_error = e

    raise RuntimeError(
        f"Model not loaded: Could not load model at {model_path}.\n"
        f"Tried compile=False, custom_objects, and h5 patch. Last error: {last_error}\n\n"
        "Fix: Re-run the patch cell in your notebook to regenerate best_tuned_model.keras"
    )

@st.cache_data
def load_comparison_data():
    if not os.path.exists(COMPARISON_CSV_PATH):
        raise FileNotFoundError(f"Comparison file not found: {COMPARISON_CSV_PATH}")
    return pd.read_csv(COMPARISON_CSV_PATH)

def prepare_sequence(series: pd.Series, input_length: int) -> np.ndarray:
    values = series.values.reshape(-1, 1)
    if len(values) >= input_length:
        window = values[-input_length:]
    else:
        padding = np.repeat(values[:1], input_length - len(values), axis=0)
        window = np.vstack([padding, values])
    return window.reshape(1, input_length, 1)

def compute_prediction_confidence(mape: float) -> str:
    if pd.isna(mape):
        return "Confidence unknown"
    if mape <= 0.05:
        return "★★★  High confidence — MAPE " + f"{mape*100:.1f}%"
    if mape <= 0.10:
        return "★★☆  Moderate confidence — MAPE " + f"{mape*100:.1f}%"
    return "★☆☆  Low confidence — MAPE " + f"{mape*100:.1f}%"

def plotly_theme():
    # Does NOT include xaxis/yaxis — callers set those individually
    # to avoid "multiple values for keyword argument" TypeError
    return dict(
        template="plotly_white",
        font=dict(family="Outfit", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4, r=4, t=40, b=4),
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def build_main_page():

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding:28px 22px 22px;border-bottom:1px solid rgba(255,255,255,0.07);">
            <div style="font-family:'Fraunces',serif;font-size:26px;font-weight:500;
                        color:#A5CF83;line-height:1.1;letter-spacing:-0.5px;">TSLA<br>Predictor</div>
            <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                        color:rgba(255,255,255,0.22);margin-top:7px;">Deep Learning Dashboard</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="padding:18px 22px 4px;font-size:9px;letter-spacing:1.8px;
                    text-transform:uppercase;color:rgba(255,255,255,0.22);">Parameters</div>
        """, unsafe_allow_html=True)

        model_choice    = st.selectbox("Model", ["Tuned LSTM", "SimpleRNN"])
        horizon         = st.selectbox("Prediction Horizon", ["1-Day", "5-Day", "10-Day"])
        lookback_window = st.slider("Lookback Window", 30, 90, 60, step=5)
        epochs_display  = st.slider("Training Epochs (display only)", 10, 100, 50, step=5)

        st.markdown("""
        <div style="height:1px;background:rgba(255,255,255,0.07);margin:18px 0 16px;"></div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="padding:0 20px 20px;">
        <div style="background:rgba(165,207,131,0.09);border:1px solid rgba(165,207,131,0.18);
                    border-radius:10px;padding:14px 16px;margin-bottom:10px;">
            <div style="font-size:9px;letter-spacing:1.3px;text-transform:uppercase;
                        color:rgba(255,255,255,0.28);">Best RMSE</div>
            <div style="font-family:'DM Mono',monospace;font-size:24px;color:#A5CF83;
                        margin-top:5px;letter-spacing:-0.5px;">18.21</div>
            <div style="font-size:11px;color:rgba(165,207,131,0.5);margin-top:3px;">Tuned LSTM · 1-Day</div>
        </div>
        <div style="background:rgba(236,182,95,0.07);border:1px solid rgba(236,182,95,0.18);
                    border-radius:10px;padding:14px 16px;">
            <div style="font-size:9px;letter-spacing:1.3px;text-transform:uppercase;
                        color:rgba(255,255,255,0.28);">Improvement</div>
            <div style="font-family:'DM Mono',monospace;font-size:24px;color:#ECB65F;
                        margin-top:5px;letter-spacing:-0.5px;">29.59%</div>
            <div style="font-size:11px;color:rgba(236,182,95,0.5);margin-top:3px;">vs SimpleRNN</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ── PAGE HEADER ───────────────────────────────────────────────────────────
    model_color = "#A5CF83" if model_choice == "Tuned LSTM" else "#ECB65F"
    hcol, bcol = st.columns([5, 1])
    with hcol:
        st.markdown("""
        <div class="page-header">
            <h1 style="font-size:34px;margin:0 0 5px;">
                Tesla Stock <em style="color:#6FA84E;font-style:italic;">Intelligence</em>
            </h1>
            <p style="color:#7A7A68;font-size:13px;margin:0;font-family:'Outfit',sans-serif;">
                NASDAQ: TSLA &nbsp;·&nbsp; SimpleRNN vs Tuned LSTM &nbsp;·&nbsp; Deep Learning Forecasting
            </p>
        </div>
        """, unsafe_allow_html=True)
    with bcol:
        st.markdown(f"""
        <div style="padding-top:32px;text-align:right;">
            <span style="background:#1A1A14;color:{model_color};
                         font-family:'DM Mono',monospace;font-size:11px;letter-spacing:0.5px;
                         padding:7px 16px;border-radius:20px;white-space:nowrap;">
                {model_choice.upper()}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── LOAD DATA ─────────────────────────────────────────────────────────────
    df = None
    load_error = None
    try:
        df = load_data()
    except Exception as e:
        load_error = str(e)

    if load_error or df is None:
        st.markdown(f"""
        <div style="background:#FFF3E0;border:1px solid #ECB65F;border-left:4px solid #E89951;
                    border-radius:10px;padding:14px 18px;margin:8px 0 20px;
                    font-family:'Outfit',sans-serif;font-size:13px;color:#5A3800;">
            <strong>Data not loaded:</strong> {load_error or 'Unknown error'}
            <br><span style="font-size:12px;color:#7A5500;margin-top:4px;display:block;">
            Place <code>TSLA.csv</code> in <code>C:\\PROJECT-3-LABMENTIX\\</code> and restart.</span>
        </div>
        """, unsafe_allow_html=True)
        return

    comparison_df = pd.DataFrame()
    try:
        comparison_df = load_comparison_data()
    except Exception:
        pass  # handled gracefully below

    model = None
    model_error = None
    model_path = SIMPLERNN_MODEL_PATH if model_choice == "SimpleRNN" else TUNED_MODEL_PATH
    try:
        model = load_prediction_model(model_path)
    except Exception as e:
        model_error = str(e)

    last_row      = df.iloc[-1]
    current_price = last_row.get("Close")
    high_price    = last_row.get("High")
    low_price     = last_row.get("Low")
    volume        = int(last_row["Volume"]) if "Volume" in df.columns else None

    # ── KPI STRIP ─────────────────────────────────────────────────────────────
    st.markdown('<div class="panel-heading">Market Snapshot</div>', unsafe_allow_html=True)

    # Accent bars via CSS injection on metric containers
    st.markdown("""
    <style>
    div[data-testid="column"]:nth-child(1) [data-testid="stMetric"] { border-top: 3px solid #A5CF83 !important; }
    div[data-testid="column"]:nth-child(2) [data-testid="stMetric"] { border-top: 3px solid #F0E76F !important; }
    div[data-testid="column"]:nth-child(3) [data-testid="stMetric"] { border-top: 3px solid #ECB65F !important; }
    div[data-testid="column"]:nth-child(4) [data-testid="stMetric"] { border-top: 3px solid #E89951 !important; }
    </style>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Price", f"${current_price:,.2f}" if current_price is not None else "N/A")
    k2.metric("Day High",      f"${high_price:,.2f}"    if high_price    is not None else "N/A")
    k3.metric("Day Low",       f"${low_price:,.2f}"     if low_price     is not None else "N/A")
    k4.metric("Volume",        f"{volume:,}"            if volume        is not None else "N/A")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── PRICE CHART ───────────────────────────────────────────────────────────
    st.markdown('<div class="panel-heading">Price History & Moving Averages</div>', unsafe_allow_html=True)

    # Styled toggle row instead of raw checkboxes
    st.markdown("""
    <div class="toggle-row">
        <span style="font-size:11px;color:#7A7A68;letter-spacing:0.5px;margin-right:4px;">Show:</span>
        <span class="ma-toggle active-sage">
            <span class="dot" style="background:#A5CF83;"></span> MA 7
        </span>
        <span class="ma-toggle active-amber1">
            <span class="dot" style="background:#ECB65F;"></span> MA 30
        </span>
        <span class="ma-toggle active-amber2">
            <span class="dot" style="background:#E89951;"></span> MA 90
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Actual checkboxes — hidden via CSS, state still drives logic
    t_col1, t_col2, t_col3, range_col = st.columns([1, 1, 1, 2])
    with t_col1: show_ma7  = st.checkbox("MA7",  value=True,  label_visibility="collapsed")
    with t_col2: show_ma30 = st.checkbox("MA30", value=True,  label_visibility="collapsed")
    with t_col3: show_ma90 = st.checkbox("MA90", value=True,  label_visibility="collapsed")
    range_option = range_col.selectbox(
        "Date Range", ["1Y", "3Y", "5Y", "All"], index=3,
        label_visibility="collapsed"
    )

    chart_df = df.copy()
    if show_ma7:  chart_df["MA7"]  = chart_df["Close"].rolling(7).mean()
    if show_ma30: chart_df["MA30"] = chart_df["Close"].rolling(30).mean()
    if show_ma90: chart_df["MA90"] = chart_df["Close"].rolling(90).mean()
    if range_option != "All":
        years = int(range_option[0])
        chart_df = chart_df[chart_df.index >= chart_df.index.max() - pd.DateOffset(years=years)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["Close"], name="Close",
        line=dict(color="#1A1A14", width=2.5), hovertemplate="$%{y:,.2f}<extra>Close</extra>"
    ))
    if show_ma7 and "MA7" in chart_df:
        fig.add_trace(go.Scatter(
            x=chart_df.index, y=chart_df["MA7"], name="MA 7",
            line=dict(color="#A5CF83", width=1.5), hovertemplate="$%{y:,.2f}<extra>MA7</extra>"
        ))
    if show_ma30 and "MA30" in chart_df:
        fig.add_trace(go.Scatter(
            x=chart_df.index, y=chart_df["MA30"], name="MA 30",
            line=dict(color="#ECB65F", width=1.5), hovertemplate="$%{y:,.2f}<extra>MA30</extra>"
        ))
    if show_ma90 and "MA90" in chart_df:
        fig.add_trace(go.Scatter(
            x=chart_df.index, y=chart_df["MA90"], name="MA 90",
            line=dict(color="#E89951", width=1.5), hovertemplate="$%{y:,.2f}<extra>MA90</extra>"
        ))
    fig.update_layout(
        **plotly_theme(),
        height=360,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0.05)", zeroline=False, title="Close Price (USD)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── PREDICTION + PERFORMANCE ──────────────────────────────────────────────
    pred_col, perf_col = st.columns([1, 1], gap="large")

    # LEFT: Prediction
    with pred_col:
        st.markdown('<div class="panel-heading">Price Prediction</div>', unsafe_allow_html=True)

        if model_error:
            st.markdown(f"""
            <div style="background:#FFF3E0;border:1px solid #ECB65F;border-left:4px solid #E89951;
                        border-radius:10px;padding:12px 16px;font-size:13px;color:#5A3800;
                        font-family:'Outfit',sans-serif;">
                <strong>Model not loaded:</strong> {model_error}
            </div>
            """, unsafe_allow_html=True)

        elif model is not None and "Close" in df.columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit_transform(df[["Close"]])

            input_len = (model.input_shape[1]
                         if isinstance(model.input_shape, tuple)
                         else model.input_shape[0][1])
            sequence   = prepare_sequence(df["Close"].iloc[-lookback_window:], input_len)
            scaled_seq = scaler.transform(sequence.reshape(-1, 1)).reshape(sequence.shape)

            predicted_price = None
            pred_err = None
            try:
                pred_scaled     = model.predict(scaled_seq, verbose=0)
                predicted_price = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            except Exception as e:
                pred_err = str(e)

            if pred_err:
                st.markdown(f"""
                <div style="background:#FFF3E0;border:1px solid #ECB65F;border-left:4px solid #E89951;
                            border-radius:10px;padding:12px 16px;font-size:13px;color:#5A3800;">
                    <strong>Prediction error:</strong> {pred_err}
                </div>
                """, unsafe_allow_html=True)

            elif predicted_price is not None:
                delta     = predicted_price - current_price if current_price else 0
                delta_pct = (delta / current_price * 100)   if current_price else 0
                arrow     = "▲" if delta >= 0 else "▼"
                chg_class = "pred-change-pos" if delta >= 0 else "pred-change-neg"

                sel = pd.DataFrame()
                if not comparison_df.empty:
                    sel = comparison_df[
                        (comparison_df["Model"]   == f"{model_choice} ({horizon})") &
                        (comparison_df["Horizon"] == horizon)
                    ]
                    if sel.empty and model_choice == "Tuned LSTM" and horizon != "1-Day":
                        sel = comparison_df[comparison_df["Model"] == "Tuned LSTM (1-Day)"]
                mape = sel.iloc[0]["MAPE"] if not sel.empty else np.nan
                conf = compute_prediction_confidence(mape)

                st.markdown(f"""
                <div class="pred-box">
                    <div class="pred-label">Predicted Close &nbsp;·&nbsp; {horizon}</div>
                    <div class="pred-price"><em>$</em>{predicted_price:,.2f}</div>
                    <div class="{chg_class}">{arrow} {abs(delta):.2f} ({delta_pct:+.2f}%)</div>
                    <div class="conf-badge">{conf}</div>
                </div>
                """, unsafe_allow_html=True)

                # Mini actual-vs-predicted chart
                pred_idx    = df.index[-1] + pd.DateOffset(days=int(horizon.split("-")[0]))
                actual_t    = df[["Close"]].tail(30).copy()
                actual_t["Type"] = "Actual"
                pred_t      = pd.DataFrame(
                    {"Date": [pred_idx], "Close": [predicted_price], "Type": ["Predicted"]}
                ).set_index("Date")
                chart_data  = pd.concat([actual_t, pred_t])

                fig2 = px.line(
                    chart_data, x=chart_data.index, y="Close", color="Type", markers=True,
                    color_discrete_map={"Actual": "#1A1A14", "Predicted": "#A5CF83"}
                )
                fig2.update_traces(
                    selector=dict(name="Predicted"),
                    line=dict(dash="dot", width=2),
                    marker=dict(size=10, symbol="circle")
                )
                fig2.update_layout(
                    **plotly_theme(),
                    height=260,
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(gridcolor="rgba(0,0,0,0.05)", zeroline=False, title="Price (USD)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown("""
            <div style="background:#F5F5F0;border-radius:10px;padding:20px 18px;
                        font-size:13px;color:#7A7A68;text-align:center;">
                Add model files to the project folder to enable predictions.
            </div>
            """, unsafe_allow_html=True)

    # RIGHT: Model Performance
    with perf_col:
        st.markdown('<div class="panel-heading">Model Performance</div>', unsafe_allow_html=True)

        if not comparison_df.empty:
            sel_row = comparison_df[
                (comparison_df["Model"]   == f"{model_choice} ({horizon})") &
                (comparison_df["Horizon"] == horizon)
            ]
            if sel_row.empty and model_choice == "Tuned LSTM" and horizon != "1-Day":
                sel_row = comparison_df[comparison_df["Model"] == "Tuned LSTM (1-Day)"]

            mse_v  = sel_row.iloc[0]["MSE"]  if not sel_row.empty else np.nan
            rmse_v = sel_row.iloc[0]["RMSE"] if not sel_row.empty else np.nan
            mae_v  = sel_row.iloc[0]["MAE"]  if not sel_row.empty else np.nan

            def fmt(v):
                if pd.isna(v):
                    return "N/A"
                return f"{v:.2f}"

            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="model-metric"><div class="mm-label">MSE</div><div class="mm-val">{fmt(mse_v)}</div></div>',  unsafe_allow_html=True)
            m2.markdown(f'<div class="model-metric"><div class="mm-label">RMSE</div><div class="mm-val">{fmt(rmse_v)}</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="model-metric"><div class="mm-label">MAE</div><div class="mm-val">{fmt(mae_v)}</div></div>',  unsafe_allow_html=True)

            st.markdown("""
            <div class="improve-bar">
                <span class="ib-label">LSTM improvement over SimpleRNN</span>
                <span class="ib-val">−29.59%</span>
            </div>
            """, unsafe_allow_html=True)

            bar_fig = px.bar(
                comparison_df, x="Model", y="RMSE", color="Horizon",
                barmode="group",
                color_discrete_sequence=["#1A1A14", "#A5CF83", "#ECB65F"]
            )
            bar_fig.update_layout(
                **plotly_theme(),
                height=300,
                bargap=0.25,
                bargroupgap=0.08,
                xaxis=dict(showgrid=False, zeroline=False, tickangle=-12),
                yaxis=dict(gridcolor="rgba(0,0,0,0.05)", zeroline=False, title="RMSE"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)")
            )
            bar_fig.update_traces(marker_line_width=0)
            st.plotly_chart(bar_fig, use_container_width=True)

        else:
            st.markdown("""
            <div style="background:#F5F5F0;border-radius:10px;padding:20px 18px;
                        font-size:13px;color:#7A7A68;text-align:center;">
                Place <code>model_comparison_results.csv</code> in the project folder to view comparisons.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── RECENT DATA TABLE ─────────────────────────────────────────────────────
    st.markdown('<div class="panel-heading">Recent Data</div>', unsafe_allow_html=True)
    st.dataframe(
        df.tail(10).style.format({
            col: "${:,.2f}" for col in ["Open","High","Low","Close","Adj Close"]
            if col in df.columns
        }),
        use_container_width=True,
        height=340
    )

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-footer">
        Built for Tesla Stock Prediction Project &nbsp;·&nbsp; Deep Learning with SimpleRNN &amp; LSTM
        &nbsp;·&nbsp; <span style="font-family:'Fraunces',serif;font-style:italic;color:#4A4A3C;">TSLA Predictor</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    build_main_page()