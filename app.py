"""
╔══════════════════════════════════════════════════════════════╗
║         NEURAL NETWORK EXPLORER — Streamlit Dashboard        ║
║  Real trained models • Interactive testing • Live charts     ║
╚══════════════════════════════════════════════════════════════╝

Run:  streamlit run app.py
Deps: pip install streamlit plotly pandas numpy scikit-learn
"""

import streamlit as st
import numpy as np
import json, os, math
from pathlib import Path

# ── Optional: plotly for richer charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.warning("Install plotly for best charts: pip install plotly")

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Neural Network Explorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS — dark, editorial, sharp
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

/* ── root */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #07070f;
    color: #f0ede6;
}
.main { background: #07070f; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

/* ── sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d1a;
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] * { color: #f0ede6 !important; }

/* ── headings */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; letter-spacing: -0.03em !important; }
h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }

/* ── metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 1.6rem !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { font-family: 'Space Mono', monospace !important; font-size: 0.65rem !important; color: #55546a !important; text-transform: uppercase; letter-spacing: 0.1em; }

/* ── buttons */
.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    background: rgba(255,255,255,0.05) !important;
    color: #f0ede6 !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
    padding: 0.4rem 1.1rem !important;
}
.stButton > button:hover { background: rgba(255,255,255,0.12) !important; border-color: rgba(255,255,255,0.3) !important; }

/* ── inputs */
.stNumberInput input, .stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #f0ede6 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stSlider > div > div > div { background: rgba(255,255,255,0.1) !important; }

/* ── tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.07);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-radius: 7px !important;
    color: #55546a !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #f0ede6 !important;
    background: rgba(255,255,255,0.08) !important;
}

/* ── code / mono */
code { font-family: 'Space Mono', monospace !important; }

/* ── divider */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── expander */
.streamlit-expanderHeader {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
}

/* ── dataframe */
.dataframe { font-family: 'Space Mono', monospace !important; font-size: 0.7rem !important; }

/* ── custom cards */
.nn-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.5rem;
}
.tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 9px;
    border-radius: 100px;
    display: inline-block;
}
.formula-box {
    background: rgba(255,208,0,0.07);
    border: 1px solid rgba(255,208,0,0.2);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #ffd000;
    letter-spacing: 0.04em;
    margin: 0.75rem 0;
}
.result-box {
    background: rgba(0,200,122,0.08);
    border: 1px solid rgba(0,200,122,0.25);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #00c87a;
    margin-top: 0.75rem;
}
.error-box {
    background: rgba(255,69,0,0.08);
    border: 1px solid rgba(255,69,0,0.25);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #ff7a50;
    margin-top: 0.75rem;
}
.info-pill {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 100px;
    padding: 2px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #9b99aa;
    margin: 2px;
}
.section-head {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #55546a;
    margin-bottom: 0.6rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD ALL MODEL DATA  (embedded inline — works on Streamlit Cloud)
# ══════════════════════════════════════════════════════════════

@st.cache_data
def load_models():
    from pathlib import Path
    base = Path(__file__).resolve().parent
    return {name: json.loads((base / f"out_{name}.json").read_text(encoding="utf-8"))
            for name in ["mse", "perceptron", "backprop", "rnn", "lstm", "cnn"]}


DATA = load_models()
st.caption("Educational datasets. CNN accuracy is measured on a held-out test set; recurrent R² uses a chronological holdout. AND/XOR scores are training-set scores. Forecasts are demonstrations.")


# ══════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Mono, monospace", color="#9b99aa", size=11),
    margin=dict(l=50, r=20, t=40, b=50),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)
NN_COLORS = {
    "cnn": "#ff4500", "rnn": "#0055ff", "bp": "#00c87a",
    "perc": "#9b00ff", "lstm": "#ffd000", "mse": "#00ddb0",
}


def styled_fig(fig):
    fig.update_layout(**PLOT_LAYOUT)
    return fig


# ══════════════════════════════════════════════════════════════
# INFERENCE ENGINES (pure numpy, from trained weights)
# ══════════════════════════════════════════════════════════════

def infer_mse(sqft: float) -> dict:
    """Linear regression: ŷ = w·x + b"""
    d = DATA["mse"]
    w, b = d["final_w"], d["final_b"]
    x = sqft / 100
    price = w * x + b
    return {"price_k": round(price, 1), "w": w, "b": b,
            "formula": f"ŷ = {w:.4f}×{x:.2f} + {b:.4f} = {price:.1f}k"}


def infer_perceptron(x1: float, x2: float) -> dict:
    """Single perceptron: ŷ = step(w₁x₁ + w₂x₂ + b)"""
    d = DATA["perceptron"]
    w = d["weights"]
    z = w["w0"] * x1 + w["w1"] * x2 + w["b"]
    out = 1 if z >= 0 else 0
    return {"output": out, "z": round(z, 4),
            "w0": w["w0"], "w1": w["w1"], "b": w["b"],
            "formula": f"z = {w['w0']}×{x1} + {w['w1']}×{x2} + ({w['b']}) = {z:.4f} → {out}"}


def infer_backprop(x1, x2):
    from models import as_arrays, sigmoid
    p = as_arrays(DATA["backprop"]["weights"])
    x = np.array([[x1, x2]])
    prob = float(sigmoid(sigmoid(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"])[0, 0])
    pred = int(prob > .5)
    return {"pred": pred, "prob": prob, "correct": pred == int(bool(x1) != bool(x2))}


def infer_rnn(window):
    from models import predict_sequence
    return {"next_value": round(predict_sequence(window, DATA["rnn"]), 4),
            "note": "Prediction from the entered window using exported RNN weights."}


def infer_lstm(prices):
    from models import predict_sequence
    return {"next_day": round(predict_sequence(prices, DATA["lstm"]), 2),
            "note": "Synthetic-series model; prediction uses the entered prices."}


def infer_cnn(idx):
    from models import predict_digit
    sample = DATA["cnn"]["test_cases"][idx]
    pred, probs = predict_digit(sample["pixels"], DATA["cnn"])
    return {"true": sample["true"], "pred": pred, "correct": pred == sample["true"],
            "pixels": sample["pixels"], "confidence": float(probs[pred])}


# # CHART BUILDERS
# ══════════════════════════════════════════════════════════════

def chart_loss_acc(nn_key, title="Training Curves"):
    key_map = {"bp": "backprop", "perc": "perceptron"}
    data_key = key_map.get(nn_key, nn_key)
    d = DATA.get(data_key)
    if not d:
        return None
    color = NN_COLORS.get(nn_key, "#ffffff")
    loss = d.get("loss_history", [])
    acc  = d.get("acc_history", [])
    epochs = list(range(1, len(loss)+1))

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Training Loss over Epochs", "Accuracy / R² (%, see evaluation notes)"))
    fig.add_trace(go.Scatter(x=epochs, y=loss, mode="lines",
        line=dict(color=color, width=2.5), fill="tozeroy",
        fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.12,)}",
        name="Loss"), row=1, col=1)
    if acc:
        fig.add_trace(go.Scatter(x=list(range(1, len(acc)+1)), y=acc, mode="lines",
            line=dict(color="#00c87a", width=2.5), fill="tozeroy",
            fillcolor="rgba(0,200,122,0.12)", name="Acc/R²"), row=1, col=2)
    fig.update_layout(**PLOT_LAYOUT, title=title, height=320)
    fig.update_xaxes(title_text="Epoch", gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def chart_mse_scatter(animate_line=True):
    d = DATA["mse"]
    fig = go.Figure()
    # residuals (faint)
    scatter_x = [p["x"] for p in d["scatter"]]
    scatter_y = [p["y"] for p in d["scatter"]]
    line_x = [p["x"] for p in d["line"]]
    line_y = [p["y"] for p in d["line"]]
    # Residual lines
    for pt in d["scatter"]:
        # interpolate line y at pt x
        for i in range(len(line_x)-1):
            if line_x[i] <= pt["x"] <= line_x[i+1]:
                t = (pt["x"]-line_x[i])/(line_x[i+1]-line_x[i])
                ly = line_y[i] + t*(line_y[i+1]-line_y[i])
                fig.add_shape(type="line", x0=pt["x"], y0=pt["y"], x1=pt["x"], y1=ly,
                              line=dict(color="rgba(255,69,0,0.25)", width=1))
                break
    # scatter
    fig.add_trace(go.Scatter(x=scatter_x, y=scatter_y, mode="markers",
        marker=dict(size=10, color="#00c87a", opacity=0.85,
                    line=dict(color="rgba(255,255,255,0.3)", width=1)),
        name="House Data"))
    # regression line
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines",
        line=dict(color="#ff4500", width=3), name=f"ŷ={d['final_w']:.3f}x+{d['final_b']:.1f}"))
    fig.update_layout(**PLOT_LAYOUT,
        title="House Price Regression — Scatter + Fitted Line",
        xaxis_title="House Size (sqft)",
        yaxis_title="Price ($k)", height=400)
    return fig


def chart_rnn_series():
    d = DATA["rnn"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["t"], y=d["actual"], mode="lines",
        line=dict(color="#00c87a", width=1.5), name="Actual", opacity=0.8))
    fig.add_trace(go.Scatter(x=d["t"], y=d["predicted"], mode="lines",
        line=dict(color="#0055ff", width=2), name="Predicted"))
    fig.add_trace(go.Scatter(x=d["forecast_t"], y=d["forecast"], mode="lines",
        line=dict(color="#ffd000", width=2, dash="dash"), name="Forecast"))
    fig.add_vrect(x0=d["forecast_t"][0], x1=d["forecast_t"][-1],
                  fillcolor="rgba(255,208,0,0.05)", line_width=0)
    fig.update_layout(**PLOT_LAYOUT,
        title="RNN Sine Wave: Actual vs Predicted + 20-step Forecast",
        xaxis_title="Time (t)", yaxis_title="Amplitude", height=380)
    return fig


def chart_lstm_stock():
    d = DATA["lstm"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["days_actual"], y=d["actual"], mode="lines",
        line=dict(color="#00c87a", width=1.5), name="Actual Price", opacity=0.8))
    fig.add_trace(go.Scatter(x=d["days_actual"], y=d["predicted"], mode="lines",
        line=dict(color="#ffd000", width=2), name="LSTM Predicted"))
    fig.add_trace(go.Scatter(x=d["forecast_days"], y=d["forecast"], mode="lines",
        line=dict(color="#ff6400", width=2, dash="dash"), name="15-day Forecast",
        fill="tonexty" if False else None))
    fig.add_vrect(x0=d["forecast_days"][0], x1=d["forecast_days"][-1],
                  fillcolor="rgba(255,100,0,0.06)", line_width=0)
    # forecast ribbon
    upper = [v + 4 for v in d["forecast"]]
    lower = [v - 4 for v in d["forecast"]]
    fig.add_trace(go.Scatter(x=d["forecast_days"]+d["forecast_days"][::-1],
        y=upper+lower[::-1], fill="toself",
        fillcolor="rgba(255,100,0,0.08)", line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, name="Forecast Band"))
    fig.update_layout(**PLOT_LAYOUT,
        title="LSTM Stock Price: Actual vs Predicted + 15-day Forecast",
        xaxis_title="Trading Day", yaxis_title="Price ($)", height=380)
    return fig


def chart_perceptron_boundary(epoch_frac=1.0):
    d = DATA["perceptron"]
    # pick boundary snapshot
    target = int(epoch_frac * d["epochs"])
    snap = d["boundaries"][0]
    for b in d["boundaries"]:
        if b["ep"] <= max(target, 1):
            snap = b
    fig = go.Figure()
    # background regions
    xs = np.linspace(-0.4, 1.5, 60)
    ys_boundary = []
    if abs(snap["ys"][1] - snap["ys"][0]) > 0.01:
        for xv in xs:
            t = (xv - snap["xs"][0]) / (snap["xs"][1] - snap["xs"][0])
            yv = snap["ys"][0] + t * (snap["ys"][1] - snap["ys"][0])
            ys_boundary.append(yv)
    # decision boundary line
    bx = [snap["xs"][0], snap["xs"][1]]
    by = [snap["ys"][0], snap["ys"][1]]
    fig.add_trace(go.Scatter(x=bx, y=by, mode="lines",
        line=dict(color="rgba(155,0,255,0.8)", width=3, dash="dash"),
        name=f"Decision boundary (ep {snap['ep']}, acc {snap['acc']}%)"))
    # data points
    colors = {"0": "#ff4500", "1": "#9b00ff"}
    for pt in d["data_points"]:
        fig.add_trace(go.Scatter(x=[pt["x"]], y=[pt["y"]], mode="markers+text",
            marker=dict(size=22, color=colors[str(pt["label"])],
                        line=dict(color="white", width=2)),
            text=[str(pt["label"])], textposition="middle center",
            textfont=dict(color="white", size=12, family="Space Mono"),
            name=f"AND={pt['label']}", showlegend=False))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono, monospace", color="#9b99aa", size=11),
        margin=dict(l=50, r=20, t=40, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        title=f"Perceptron Decision Boundary — Epoch {snap['ep']} · Acc {snap['acc']}%",
        xaxis=dict(range=[-0.4, 1.5], title="x₁", gridcolor="rgba(255,255,255,0.05)",
                   linecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(range=[-0.4, 1.5], title="x₂", gridcolor="rgba(255,255,255,0.05)",
                   linecolor="rgba(255,255,255,0.1)"),
        height=400)
    return fig


def chart_backprop_heatmap(snap_idx=2):
    d = DATA["backprop"]
    snap = d["snapshots"][min(snap_idx, len(d["snapshots"])-1)]
    xs, ys, z = snap["boundary"]["xs"], snap["boundary"]["ys"], snap["boundary"]["z"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"Decision Surface (Epoch {snap['ep']})", "XOR Data Points"))
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=z,
        colorscale=[[0,"rgba(255,69,0,0.7)"],[0.5,"rgba(20,20,40,0.5)"],[1,"rgba(0,200,122,0.7)"]],
        showscale=True, colorbar=dict(title="P(y=1)", len=0.6)),
        row=1, col=1)
    lbl_colors = ["#ff4500","#00c87a","#00c87a","#ff4500"]
    for i, pt in enumerate(d["data_points"]):
        mx = xs[0] + pt["x"]*(xs[-1]-xs[0])
        my = ys[0] + pt["y"]*(ys[-1]-ys[0])
        fig.add_trace(go.Scatter(x=[mx], y=[my], mode="markers+text",
            marker=dict(size=20, color=lbl_colors[i], line=dict(color="white",width=2)),
            text=[str(pt["label"])], textposition="middle center",
            textfont=dict(color="white", size=11, family="Space Mono"),
            showlegend=False), row=1, col=1)
    # gradient norm
    gn = d["grad_norm_history"]
    fig.add_trace(go.Scatter(x=list(range(1,len(gn)+1)), y=gn, mode="lines",
        line=dict(color="#00c87a", width=2), name="‖∇W‖"), row=1, col=2)
    fig.update_layout(**PLOT_LAYOUT,
        title=f"Backprop XOR — Epoch {snap['ep']} · Acc {snap['acc']}%",
        height=380)
    return fig


def chart_cnn_confusion():
    d = DATA["cnn"]
    cm = d["confusion_matrix"]
    labels = list(range(10))
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0,"#07070f"],[0.3,"rgba(0,85,255,0.4)"],[1,"rgba(0,85,255,0.95)"]],
        text=cm, texttemplate="%{text}",
        textfont=dict(size=11, family="Space Mono"),
        showscale=True))
    fig.update_layout(**PLOT_LAYOUT,
        title=f"CNN Confusion Matrix — Test Acc {d['final_acc']}%",
        xaxis_title="Predicted", yaxis_title="True", height=420)
    return fig


def chart_cnn_filters(filters):
    """Render 8 conv filters as small heatmaps."""
    names = ["Horiz. edge","Vert. edge","Laplacian","Sharpening","Learned 5","Learned 6","Learned 7","Learned 8"]
    fig = make_subplots(rows=2, cols=4,
        subplot_titles=names[:len(filters)],
        horizontal_spacing=0.08, vertical_spacing=0.15)
    for i, f in enumerate(filters):
        r, c = i//4+1, i%4+1
        fig.add_trace(go.Heatmap(z=f, colorscale="Greys", showscale=False), row=r, col=c)
    fig.update_layout(**PLOT_LAYOUT, title="Learned Conv Filters (3×3)", height=280)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def chart_cnn_feature_maps(feature_maps):
    names = ["Horiz. edges","Vert. edges","Laplacian","Sharpening"]
    fig = make_subplots(rows=1, cols=4,
        subplot_titles=names[:len(feature_maps)],
        horizontal_spacing=0.06)
    for i, fm in enumerate(feature_maps):
        fig.add_trace(go.Heatmap(z=fm,
            colorscale=[[0,"#07070f"],[1,"#ff4500"]], showscale=False), row=1, col=i+1)
    fig.update_layout(**PLOT_LAYOUT, title="Feature Maps after Conv (6×6 output)", height=220)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def render_digit(pixels, title="", size=8):
    """Render an 8×8 digit image."""
    img = np.array(pixels).reshape(size, size)
    fig = go.Figure(go.Heatmap(z=img, colorscale="Greys_r", showscale=False))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5,r=5,t=30,b=5), height=140,
        title=dict(text=title, font=dict(size=11, family="Space Mono", color="#9b99aa")),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False))
    return fig


# ══════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;letter-spacing:-0.03em;line-height:1.1;">
            Neural Network<br>Explorer
        </div>
        <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#55546a;margin-top:0.4rem;letter-spacing:0.1em;text-transform:uppercase;">
            // real trained models
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Select Model</div>', unsafe_allow_html=True)

    nn_options = {
        "⚡ MSE — Linear Regression":      "mse",
        "○  Perceptron — AND Gate":         "perc",
        "◈  Backprop — XOR MLP":            "bp",
        "⟳  RNN — Sine Forecasting":        "rnn",
        "⊞  LSTM — Stock Prediction":       "lstm",
        "⊡  CNN — Digit Classification":    "cnn",
    }
    choice = st.radio("Select a model", list(nn_options.keys()), label_visibility="collapsed")
    nn_key = nn_options[choice]

    st.markdown("---")
    st.markdown('<div class="section-head">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#55546a;line-height:1.8;">
    Small educational models trained in NumPy<br>
    in pure NumPy — no PyTorch.<br><br>
    Each model exports real weights<br>
    used for live inference here.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    # Model status
    st.markdown('<div class="section-head">Model Status</div>', unsafe_allow_html=True)
    for label, key in [("MSE","mse"),("Perceptron","perceptron"),("Backprop","backprop"),
                        ("RNN","rnn"),("LSTM","lstm"),("CNN","cnn")]:
        status = "✓" if DATA.get(key) else "✗"
        color = "#00c87a" if DATA.get(key) else "#ff4500"
        st.markdown(f'<span style="font-family:Space Mono,monospace;font-size:0.65rem;color:{color};">{status} {label}</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE RENDERING PER MODEL
# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# 1. MSE — LINEAR REGRESSION
# ─────────────────────────────────────────────────────────────
if nn_key == "mse":
    d = DATA["mse"]
    st.markdown("""
    <h1>MSE · Linear Regression</h1>
    <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#55546a;margin-top:-0.3rem;">
    House Price Prediction · trained via gradient descent · 60 epochs
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">MSE = (1/n) · Σ(yᵢ − ŷᵢ)² &nbsp;&nbsp;|&nbsp;&nbsp; ŷ = w·x + b &nbsp;&nbsp;|&nbsp;&nbsp; w ← w − α·∂MSE/∂w</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final MSE Loss", f"{d['final_loss']:.5f}")
    col2.metric("Weight (w)", f"{d['final_w']:.4f}")
    col3.metric("Bias (b)", f"{d['final_b']:.2f}")
    col4.metric("Training Epochs", str(d["epochs"]))

    tab1, tab2, tab3 = st.tabs(["📊  Scatter + Regression", "📉  Loss Curve", "🧪  Test Model"])

    with tab1:
        if HAS_PLOTLY:
            st.plotly_chart(chart_mse_scatter(), width="stretch")
        st.markdown(f"""
        <div class="nn-card">
        <div class="section-head">Model Equation</div>
        <code>ŷ = {d['final_w']:.4f} × (sqft/100) + {d['final_b']:.4f}</code><br><br>
        <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#55546a;">
        The gradient descent algorithm iterated {d['epochs']} times, adjusting w and b to minimize
        the mean squared difference between predicted and actual house prices.
        Red lines show residuals (prediction errors).
        </div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        if HAS_PLOTLY:
            fig = chart_loss_acc("mse", "MSE Loss over Training")
            st.plotly_chart(fig, width="stretch")
        with st.expander("View loss history data"):
            import math
            cols = st.columns(6)
            for i, v in enumerate(d["loss_history"]):
                cols[i%6].metric(f"Ep {i+1}", f"{v:.5f}")

    with tab3:
        st.markdown('<div class="section-head">Live Inference — Predict House Price</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            sqft = st.slider("House Size (sqft)", min_value=500, max_value=6000,
                              value=2000, step=50)
            sqft_input = st.number_input("Or type exact sqft:", min_value=100.0,
                                          max_value=10000.0, value=float(sqft), step=50.0)
            final_sqft = sqft_input
        with c2:
            if st.button("🔮  Predict Price", key="mse_predict"):
                result = infer_mse(final_sqft)
                st.markdown(f"""
                <div class="result-box">
                <b>Predicted Price: ${result['price_k']}k</b><br><br>
                <code>{result['formula']}</code>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("**Precomputed examples:**")
            for tc in d["test_cases"]:
                st.markdown(f'<span class="info-pill">{tc["sqft_100"]*100} sqft → ${tc["predicted_price_k"]}k</span>', unsafe_allow_html=True)
        # auto-predict on slider
        result = infer_mse(final_sqft)
        st.markdown(f"""
        <div class="result-box" style="margin-top:1rem;">
        <b>Live: {final_sqft:.0f} sqft → ${result['price_k']}k</b>
        &nbsp;&nbsp;<span style="opacity:.5">|</span>&nbsp;&nbsp;
        <code>ŷ = {result['w']:.4f} × {final_sqft/100:.2f} + {result['b']:.2f}</code>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 2. PERCEPTRON — AND GATE
# ─────────────────────────────────────────────────────────────
elif nn_key == "perc":
    d = DATA["perceptron"]
    st.markdown("""
    <h1>Perceptron · AND Gate</h1>
    <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#55546a;margin-top:-0.3rem;">
    Binary linear classifier trained on the four AND-gate inputs
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">ŷ = step(w₁·x₁ + w₂·x₂ + b) &nbsp;&nbsp;|&nbsp;&nbsp; wᵢ ← wᵢ + α·(y − ŷ)·xᵢ</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Accuracy", f"{d['acc_history'][-1]:.1f}%")
    col2.metric("w₁", f"{d['weights']['w0']:.4f}")
    col3.metric("w₂", f"{d['weights']['w1']:.4f}")
    col4.metric("Bias b", f"{d['weights']['b']:.4f}")

    tab1, tab2, tab3 = st.tabs(["📊  Decision Boundary", "📉  Training Curves", "🧪  Test Model"])

    with tab1:
        ep_frac = st.slider("Epoch progress", 0.0, 1.0, 1.0, 0.05, key="perc_ep")
        if HAS_PLOTLY:
            st.plotly_chart(chart_perceptron_boundary(ep_frac), width="stretch")
        st.markdown("""
        <div class="nn-card">
        <div class="section-head">How it works</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#55546a;line-height:1.8;">
        The perceptron draws a linear decision boundary in x₁–x₂ space.
        Points above the line are classified as AND=1, below as AND=0.
        Drag the slider to watch the boundary converge to the correct position.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        if HAS_PLOTLY:
            fig = chart_loss_acc("perc", "Perceptron Training")
            st.plotly_chart(fig, width="stretch")

    with tab3:
        st.markdown('<div class="section-head">Live Inference — AND Gate</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        x1_val = c1.selectbox("Input x₁", [0, 1], index=1)
        x2_val = c2.selectbox("Input x₂", [0, 1], index=1)
        result = infer_perceptron(x1_val, x2_val)
        correct = (result["output"] == (x1_val and x2_val))
        st.markdown(f"""
        <div class="{'result-box' if correct else 'error-box'}">
        <b>AND({x1_val}, {x2_val}) = {result['output']}</b>
        &nbsp; {'✓ Correct' if correct else '✗ Wrong'}<br><br>
        <code>{result['formula']}</code>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Truth table:**")
        cols = st.columns(4)
        for i, tc in enumerate(d["test_cases"]):
            ok = tc["predicted"] == tc["expected"]
            cols[i].markdown(f"""<div class="nn-card" style="text-align:center;padding:.75rem;">
            <div style="font-family:Space Mono,monospace;font-size:.65rem;color:#55546a;">({tc['inputs'][0]},{tc['inputs'][1]})</div>
            <div style="font-size:1.4rem;font-weight:800;color:{'#00c87a' if ok else '#ff4500'};">{tc['predicted']}</div>
            <div style="font-family:Space Mono,monospace;font-size:.6rem;color:#55546a;">exp: {tc['expected']}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 3. BACKPROP — XOR
# ─────────────────────────────────────────────────────────────
elif nn_key == "bp":
    d = DATA["backprop"]
    st.markdown("""
    <h1>Backpropagation · XOR MLP</h1>
    <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#55546a;margin-top:-0.3rem;">
    2-layer MLP · chain rule gradients · non-linearly separable problem
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">∂L/∂w = δ·xᵀ &nbsp;|&nbsp; δₗ = (δₗ₊₁·Wₗ₊₁ᵀ) ⊙ σ′(zₗ) &nbsp;|&nbsp; w ← w − α·∂L/∂w</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Accuracy", f"{d['acc_history'][-1]:.1f}%")
    col2.metric("Final Loss", f"{d['loss_history'][-1]:.6f}")
    col3.metric("Architecture", "2 → 4 → 1")

    tab1, tab2, tab3 = st.tabs(["🗺️  Decision Surface", "📉  Training + Gradients", "🧪  Test Model"])

    with tab1:
        snap_choice = st.select_slider("Snapshot", options=["Epoch 1","Epoch 50","Epoch 100"],
                                        value="Epoch 100")
        snap_idx = {"Epoch 1": 0, "Epoch 50": 1, "Epoch 100": 2}[snap_choice]
        if HAS_PLOTLY:
            st.plotly_chart(chart_backprop_heatmap(snap_idx), width="stretch")

    with tab2:
        if HAS_PLOTLY:
            st.plotly_chart(chart_loss_acc("bp", "Backprop Training Curves"), width="stretch")
        st.markdown("""
        <div class="nn-card">
        <div class="section-head">Why XOR is hard</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#55546a;line-height:1.8;">
        XOR is not linearly separable — no single line can divide the classes.
        The hidden layer creates a non-linear transformation that makes the classes separable.
        The right chart shows gradient norm ‖∇W‖ — it starts high and stabilises as the network converges.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-head">Live Inference — XOR Classification</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        x1_val = c1.selectbox("Input x₁", [0, 1], index=0, key="bp_x1")
        x2_val = c2.selectbox("Input x₂", [0, 1], index=1, key="bp_x2")
        result = infer_backprop(x1_val, x2_val)
        if "error" in result:
            st.markdown(f'<div class="error-box">{result["error"]}</div>', unsafe_allow_html=True)
        else:
            expected = int(x1_val) ^ int(x2_val)
            ok = result["pred"] == expected
            prob_str = f" (prob={result['prob']:.4f})" if result["prob"] is not None else ""
            st.markdown(f"""
            <div class="{'result-box' if ok else 'error-box'}">
            <b>XOR({x1_val}, {x2_val}) = {result['pred']}</b>{prob_str}
            &nbsp; {'✓ Correct' if ok else '✗ Incorrect'}<br>
            <span style="opacity:.6">Expected: {expected}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("**Full truth table:**")
        cols2 = st.columns(4)
        for i, tc in enumerate(d["test_cases"]):
            ok2 = tc["pred"] == tc["expected"]
            p = f"{tc['prob']:.3f}" if tc.get("prob") else "—"
            cols2[i].markdown(f"""<div class="nn-card" style="text-align:center;padding:.75rem;">
            <div style="font-family:Space Mono,monospace;font-size:.65rem;color:#55546a;">({tc['inputs'][0]},{tc['inputs'][1]})</div>
            <div style="font-size:1.4rem;font-weight:800;color:{'#00c87a' if ok2 else '#ff4500'};">{tc['pred']}</div>
            <div style="font-family:Space Mono,monospace;font-size:.6rem;color:#55546a;">p={p}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 4. RNN — SINE WAVE
# ─────────────────────────────────────────────────────────────
elif nn_key == "rnn":
    d = DATA["rnn"]
    st.markdown("""
    <h1>RNN · Sine Wave Forecasting</h1>
    <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#55546a;margin-top:-0.3rem;">
    Elman RNN · hidden size 8 · chronological holdout · synthetic sine series
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">hₜ = tanh(Wₓ·xₜ + Wₕ·hₜ₋₁ + b) &nbsp;&nbsp;|&nbsp;&nbsp; ŷ = Wᵧ·hₜ + bᵧ</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Holdout R²", f"{d['acc_history'][-1]:.1f}%")
    col2.metric("Final Loss", f"{d['loss_history'][-1]:.6f}")
    col3.metric("Hidden Units", "8")
    col4.metric("Forecast Steps", "20")

    tab1, tab2, tab3 = st.tabs(["📈  Time Series", "📉  Training Curves", "🧪  Test Model"])

    with tab1:
        if HAS_PLOTLY:
            st.plotly_chart(chart_rnn_series(), width="stretch")
        n_show = st.slider("Show first N timesteps", 20, 150, 150, key="rnn_n")
        if HAS_PLOTLY and n_show < 150:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=d["t"][:n_show], y=d["actual"][:n_show],
                line=dict(color="#00c87a", width=1.5), name="Actual"))
            fig2.add_trace(go.Scatter(x=d["t"][:n_show], y=d["predicted"][:n_show],
                line=dict(color="#0055ff", width=2), name="Predicted"))
            fig2.update_layout(**PLOT_LAYOUT, height=280, title=f"First {n_show} Timesteps")
            st.plotly_chart(fig2, width="stretch")

    with tab2:
        if HAS_PLOTLY:
            st.plotly_chart(chart_loss_acc("rnn", "RNN Training Curves"), width="stretch")

    with tab3:
        st.markdown('<div class="section-head">Live Inference — Next Value Prediction</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="nn-card">
        <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#55546a;line-height:1.8;">
        Enter 10 comma-separated values as a sequence window. The RNN predicts the next value.
        </div>
        </div>
        """, unsafe_allow_html=True)
        window_str = st.text_input("10-value window (comma-separated)",
                                    value=",".join(str(v) for v in d["test_window"]))
        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("🔮  Predict Next", key="rnn_pred"):
                try:
                    vals = [float(x.strip()) for x in window_str.split(",")]
                    result = infer_rnn(vals)
                    st.markdown(f"""
                    <div class="result-box">
                    <b>Predicted next value: {result['next_value']}</b><br>
                    <span style="opacity:.6;font-size:.75rem;">{result['note']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                except (ValueError, TypeError):
                    st.markdown('<div class="error-box">Invalid input — use 10 comma-separated numbers</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="result-box">
            <b>Model's answer: {d['test_result']}</b><br>
            Input window: last ten observations of the synthetic series
            </div>
            """, unsafe_allow_html=True)
        st.markdown("**20-step Forecast:**")
        if HAS_PLOTLY:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=d["forecast_t"], y=d["forecast"], mode="lines+markers",
                line=dict(color="#ffd000", width=2.5), marker=dict(size=6),
                name="Forecast"))
            fig3.update_layout(**PLOT_LAYOUT, height=220, title="RNN 20-Step Lookahead")
            st.plotly_chart(fig3, width="stretch")


# ─────────────────────────────────────────────────────────────
# 5. LSTM — STOCK PREDICTION
# ─────────────────────────────────────────────────────────────
elif nn_key == "lstm":
    d = DATA["lstm"]
    st.markdown("""
    <h1>LSTM · Synthetic Price Forecasting</h1>
    <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#55546a;margin-top:-0.3rem;">
    NumPy LSTM · all gates trained with BPTT · synthetic price series
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">fₜ=σ(Wf·[hₜ₋₁,xₜ]+bf) &nbsp;|&nbsp; iₜ=σ(Wi·[hₜ₋₁,xₜ]+bi) &nbsp;|&nbsp; cₜ=fₜ⊙cₜ₋₁+iₜ⊙g̃ₜ &nbsp;|&nbsp; hₜ=oₜ⊙tanh(cₜ)</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Next Forecast", f"${d['test_output']}")
    col2.metric("Price Range", f"${d['price_range'][0]:.0f}–${d['price_range'][1]:.0f}")
    col3.metric("LSTM Hidden", "12")
    col4.metric("Forecast Days", "15")

    tab1, tab2, tab3 = st.tabs(["📈  Stock Chart", "📉  Training Curves", "🧪  Test Model"])

    with tab1:
        if HAS_PLOTLY:
            st.plotly_chart(chart_lstm_stock(), width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div class="nn-card">
            <div class="section-head">LSTM Gates</div>
            <div style="font-family:'Space Mono',monospace;font-size:.65rem;color:#55546a;line-height:1.8;">
            <b style="color:#f0ede6;">Forget gate fₜ</b> — decides what to discard<br>
            <b style="color:#f0ede6;">Input gate iₜ</b> — decides what new info to store<br>
            <b style="color:#f0ede6;">Cell state cₜ</b> — the long-term memory<br>
            <b style="color:#f0ede6;">Output gate oₜ</b> — controls hidden state output
            </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="nn-card">
            <div class="section-head">Forecast Summary</div>
            <div style="font-family:'Space Mono',monospace;font-size:.65rem;color:#55546a;line-height:1.8;">
            Days 150–164: <b style="color:#f0ede6;">${min(d['forecast']):.2f} – ${max(d['forecast']):.2f}</b><br>
            Trend: <b style="color:#f0ede6;">{'📈 Up' if d['forecast'][-1] > d['forecast'][0] else '📉 Down'}</b><br>
            Last actual day price: <b style="color:#f0ede6;">${d['actual'][-1]}</b><br>
            Day 150 forecast: <b style="color:#ff6400;">${d['forecast'][0]}</b>
            </div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        if HAS_PLOTLY:
            st.plotly_chart(chart_loss_acc("lstm", "LSTM Training Curves"), width="stretch")

    with tab3:
        st.markdown('<div class="section-head">Live Inference — Next Day Price</div>', unsafe_allow_html=True)
        prices_str = st.text_input("Last 10 closing prices (comma-separated, $)",
                                    value=",".join(str(v) for v in d["test_input"]))
        if st.button("🔮  Predict Tomorrow", key="lstm_pred"):
            try:
                prices = [float(x.strip()) for x in prices_str.split(",")]
                result = infer_lstm(prices)
                st.markdown(f"""
                <div class="result-box">
                <b>Predicted next-day price: ${result['next_day']}</b><br>
                <span style="opacity:.6;font-size:.75rem;">{result['note']}</span>
                </div>
                """, unsafe_allow_html=True)
            except (ValueError, TypeError):
                st.markdown('<div class="error-box">Enter exactly 10 comma-separated prices</div>', unsafe_allow_html=True)
        st.markdown("**15-day Forecast Detail:**")
        fcols = st.columns(5)
        for i, (day, price) in enumerate(zip(d["forecast_days"], d["forecast"])):
            change = price - d["actual"][-1]
            fcols[i%5].markdown(f"""<div class="nn-card" style="text-align:center;padding:.6rem;">
            <div style="font-family:Space Mono,monospace;font-size:.6rem;color:#55546a;">Day {day}</div>
            <div style="font-size:1rem;font-weight:800;">${price:.1f}</div>
            <div style="font-family:Space Mono,monospace;font-size:.6rem;color:{'#00c87a' if change>=0 else '#ff4500'};">
            {'▲' if change>=0 else '▼'}{abs(change):.1f}
            </div></div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 6. CNN — DIGIT CLASSIFICATION
# ─────────────────────────────────────────────────────────────
elif nn_key == "cnn":
    d = DATA["cnn"]
    st.markdown("""
    <h1>CNN · Digit Classification</h1>
    <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#55546a;margin-top:-0.3rem;">
    sklearn digits (8×8) · learned 3×3 convolution filters · Adam training
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula-box">f(x) = softmax(W·flatten(MaxPool(ReLU(W★x+b))+b))</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy", f"{d['final_acc']}%")
    col2.metric("Conv Filters", "8 × (3×3)")
    col3.metric("Training Epochs", str(d["epochs"]))
    col4.metric("Classes", "10 (0–9)")

    tab1, tab2, tab3, tab4 = st.tabs(["🔢  Confusion Matrix", "🎨  Learned Filters", "🗺️  Feature Maps", "🧪  Test Model"])

    with tab1:
        if HAS_PLOTLY:
            st.plotly_chart(chart_cnn_confusion(), width="stretch")
        if HAS_PLOTLY:
            st.plotly_chart(chart_loss_acc("cnn", "CNN Training: Loss + Accuracy"), width="stretch")

    with tab2:
        if HAS_PLOTLY:
            st.plotly_chart(chart_cnn_filters(d["filters"]), width="stretch")
        st.markdown("""
        <div class="nn-card">
        <div class="section-head">Filter Interpretations</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem;">
        <div style="font-family:'Space Mono',monospace;font-size:.65rem;color:#55546a;line-height:1.7;">
        <b style="color:#f0ede6;">Filters 1–4</b> — Learned convolution kernels<br>
        No fixed edge templates are inserted<br>
        Interpretation requires inspecting activations<br>
        All filters are optimized during training
        </div>
        <div style="font-family:'Space Mono',monospace;font-size:.65rem;color:#55546a;line-height:1.7;">
        <b style="color:#f0ede6;">Filters 5–8</b> — Learned by Adam from data.<br>
        These patterns emerged automatically<br>
        to minimise cross-entropy loss on<br>
        handwritten digit images.
        </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        if HAS_PLOTLY:
            st.plotly_chart(chart_cnn_feature_maps(d["feature_maps"]), width="stretch")
        st.markdown("""
        <div class="nn-card">
        <div style="font-family:'Space Mono',monospace;font-size:.68rem;color:#55546a;line-height:1.8;">
        Feature maps show the output of each filter applied to a sample digit image (8×8 → 6×6 after 3×3 conv with no padding).
        Brighter areas indicate where that filter detected its pattern most strongly.
        </div>
        </div>
        """, unsafe_allow_html=True)
        # Show sample images
        st.markdown('<div class="section-head">Sample Images</div>', unsafe_allow_html=True)
        img_cols = st.columns(5)
        for i, img in enumerate(d["sample_images"]):
            with img_cols[i]:
                if HAS_PLOTLY:
                    label = f"True:{img['true']} Pred:{img['pred']} {'✓' if img['correct'] else '✗'}"
                    st.plotly_chart(render_digit(img["pixels"], label), width="stretch")

    with tab4:
        st.markdown('<div class="section-head">Live Inference — Classify Sample Digit</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            sample_idx = st.number_input("Sample index (0–9)", min_value=0, max_value=9,
                                          value=0, step=1)
            result = infer_cnn(int(sample_idx))
        with c2:
            if result["pixels"] and HAS_PLOTLY:
                label = f"True: {result['true']} | Predicted: {result['pred']} | {'✓ Correct' if result['correct'] else '✗ Wrong'}"
                st.plotly_chart(render_digit(result["pixels"], label), width="stretch")
        status = "result-box" if result["correct"] else "error-box"
        st.markdown(f"""
        <div class="{status}">
        <b>True label: {result['true']} → CNN predicts: {result['pred']}</b>
        &nbsp;&nbsp; {'✓ Correct classification' if result['correct'] else '✗ Misclassification'}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**All 10 test samples:**")
        tcols = st.columns(5)
        for i, tc in enumerate(d["test_cases"]):
            ok = tc["correct"]
            tcols[i%5].markdown(f"""<div class="nn-card" style="text-align:center;padding:.6rem;">
            <div style="font-family:Space Mono,monospace;font-size:.6rem;color:#55546a;">Sample {i}</div>
            <div style="font-size:1.3rem;font-weight:800;color:{'#00c87a' if ok else '#ff4500'};">{tc['pred']}</div>
            <div style="font-family:Space Mono,monospace;font-size:.6rem;color:#55546a;">true: {tc['true']}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#55546a;text-align:center;padding:.5rem 0;">
Small educational models trained in NumPy in pure NumPy · No PyTorch · No TensorFlow<br>
MSE · Perceptron · Backprop · RNN · LSTM · CNN · Real gradient descent · Real inference
</div>
""", unsafe_allow_html=True)
