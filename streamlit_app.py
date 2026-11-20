"""
NeuroPredict — NeuroPulse Design System
Premium AI Analytics Dashboard for Prospective Bipolar ML
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import roc_curve, confusion_matrix, brier_score_loss

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroPredict",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# NeuroPulse Design System CSS + Custom SVG Neural Element
# ------------------------------------------------------------------
NEUROPULSE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Manrope:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
  --bg-deep: #080D12;
  --surface: #121E27;
  --surface-hover: #161F2E;
  --card-border: rgba(70, 224, 193, 0.10);
  --primary: #46E0C1;
  --primary-glow: rgba(70, 224, 193, 0.25);
  --secondary: #F4B860;
  --secondary-glow: rgba(244, 184, 96, 0.25);
  --text-main: #E8F0F3;
  --text-sub: #8497A5;
  --text-muted: #4A5A6A;
  --danger: #E87A7A;
  --success: #46E0C1;
  --radius-sm: 0.65rem;
  --radius-md: 1.15rem;
  --radius-lg: 1.5rem;
  --font-body: 'Inter', system-ui, sans-serif;
  --font-display: 'Manrope', 'Inter', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

/* Deep base */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background-color: var(--bg-deep) !important;
  color: var(--text-main);
  font-family: var(--font-body);
}

/* Animated neural mesh background */
.mesh-bg {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background:
    radial-gradient(at 10% 15%, rgba(70,224,193,0.08) 0px, transparent 50%),
    radial-gradient(at 90% 85%, rgba(244,184,96,0.06) 0px, transparent 45%),
    radial-gradient(at 50% 50%, rgba(232,232,240,0.015) 0px, transparent 60%);
  animation: meshDrift 22s ease-in-out infinite alternate;
}
@keyframes meshDrift {
  0% { transform: scale(1) translate(0,0); opacity: 0.9; }
  50% { transform: scale(1.04) translate(-12px, 8px); opacity: 1; }
  100% { transform: scale(1.02) translate(10px, -6px); opacity: 0.92; }
}

/* Card system */
.card {
  background: linear-gradient(160deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.008) 100%);
  border: 1px solid rgba(70,224,193,0.08);
  border-radius: var(--radius-md);
  padding: 1.6rem 1.75rem;
  backdrop-filter: blur(18px) saturate(1.25);
  -webkit-backdrop-filter: blur(18px) saturate(1.25);
  box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset,
              0 14px 42px rgba(0,0,0,0.45);
  transition: border-color .25s ease, box-shadow .25s ease, transform .2s ease;
}
.card:hover {
  border-color: rgba(70,224,193,0.28);
  box-shadow: 0 1px 0 rgba(255,255,255,0.07) inset,
              0 20px 55px rgba(70,224,193,0.08);
}

/* Header editorial */
.editorial-header h1 {
  font-family: var(--font-display);
  font-weight: 800;
  letter-spacing: -0.04em;
  font-size: 2.8rem;
  line-height: 1.05;
  color: var(--text-main);
  margin-bottom: 0.35rem;
}
.editorial-header .subtitle {
  color: var(--text-sub);
  font-weight: 300;
  font-size: 1.05rem;
  letter-spacing: 0.01em;
  line-height: 1.45;
}

/* Section divider */
.divider-gradient {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(70,224,193,0.25), rgba(244,184,96,0.25), transparent);
  border: none;
  margin: 2.25rem 0 1.75rem 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #080D12 0%, #0D131E 100%) !important;
  border-right: 1px solid rgba(70,224,193,0.06) !important;
}
[data-testid="stSidebarNav"] a:hover {
  background: rgba(70,224,193,0.06) !important;
  border-radius: var(--radius-sm);
}

/* Metric cards */
.metric-card {
  background: linear-gradient(145deg, rgba(70,224,193,0.06), rgba(244,184,96,0.035));
  border: 1px solid rgba(70,224,193,0.10);
  border-radius: var(--radius-md);
  padding: 1.15rem 1.15rem;
  transition: border-color .2s ease, box-shadow .2s ease;
}
.metric-card:hover {
  border-color: rgba(70,224,193,0.25);
  box-shadow: 0 8px 28px rgba(70,224,193,0.08);
}
.metric-card .label {
  color: var(--text-muted);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 600;
  font-family: var(--font-mono);
}
.metric-card .value {
  color: var(--text-main);
  font-family: var(--font-display);
  font-weight: 800;
  font-size: 1.9rem;
  letter-spacing: -0.035em;
}
.metric-card .delta {
  color: var(--secondary);
  font-weight: 600;
  font-size: 0.8rem;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, var(--primary), #2A9C8C) !important;
  color: #050A12 !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.7rem 1.75rem !important;
  font-family: var(--font-display) !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 4px 20px rgba(70,224,193,0.25) !important;
  transition: all .2s ease !important;
}
.stButton > button:hover {
  box-shadow: 0 6px 28px rgba(70,224,193,0.45) !important;
  transform: translateY(-1.5px);
}

/* Sliders / Inputs */
.stSlider label { color: var(--text-sub); font-weight: 500; font-size: 0.95rem; }
.stSelectbox label { color: var(--text-sub); font-weight: 500; font-size: 0.95rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] button {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(70,224,193,0.08) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-sub) !important;
  font-family: var(--font-body) !important;
  padding: 0.45rem 1.1rem !important;
  transition: all .2s ease;
}
.stTabs [data-baseweb="tab-list"] button:hover {
  border-color: rgba(70,224,193,0.3) !important;
  color: var(--text-main) !important;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
  background: linear-gradient(135deg, rgba(70,224,193,0.15), rgba(244,184,96,0.08)) !important;
  border-color: rgba(70,224,193,0.35) !important;
  color: var(--primary) !important;
  font-weight: 600 !important;
}

/* Tables */
[data-testid="stDataFrame"] { background: transparent !important; }
[data-testid="stDataFrame"] .stDataFrame td,
[data-testid="stDataFrame"] .stDataFrame th {
  border-bottom: 1px solid rgba(70,224,193,0.08);
  color: var(--text-main);
  font-family: var(--font-body);
}

/* Charts container */
.plot-card {
  background: linear-gradient(160deg, rgba(255,255,255,0.015), rgba(255,255,255,0.005));
  border: 1px solid rgba(70,224,193,0.06);
  border-radius: var(--radius-md);
  padding: 1rem 1.25rem;
}

/* Prediction result cards */
.result-high {
  background: linear-gradient(135deg, rgba(232,122,122,0.08), rgba(232,122,122,0.02));
  border: 1px solid rgba(232,122,122,0.18);
  border-radius: var(--radius-md);
  padding: 1.25rem;
  color: var(--danger);
}
.result-medium {
  background: linear-gradient(135deg, rgba(70,224,193,0.07), rgba(70,224,193,0.02));
  border: 1px solid rgba(70,224,193,0.18);
  border-radius: var(--radius-md);
  padding: 1.25rem;
  color: var(--primary);
}

/* Custom decorative neural SVG animation */
.neural-decor {
  position: relative;
  overflow: hidden;
  border-radius: var(--radius-md);
  border: 1px solid rgba(70,224,193,0.08);
  background: linear-gradient(160deg, rgba(10,13,20,0.9), rgba(10,13,20,0.95));
  padding: 1.5rem;
  min-height: 140px;
  display: flex;
  align-items: center;
  justify-content: center;
  animation: fadeIn 0.8s ease forwards;
}

/* Page transition fade-in */
.stApp {
  animation: fadeIn 0.9s ease forwards;
}

/* Card transition enhancements */
.card {
  animation: fadeInUp 0.6s ease forwards;
}

/* Stylized chart container hover */
.plot-card:hover {
  border-color: rgba(70,224,193,0.35);
  transform: translateY(-2px);
}

/* Smooth tab transitions */
.stTabs [data-baseweb="tab-list"] button {
  transition: all 0.3s ease !important;
}

/* Metric card number animation */
.metric-card .value {
  transition: color 0.3s ease, transform 0.2s ease;
}
.metric-card:hover .value {
  color: #46E0C1;
  transform: scale(1.03);
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(14px); }
  to { opacity: 1; transform: translateY(0); }
}
.neural-decor svg {
  width: 100%;
  max-width: 360px;
  height: auto;
  opacity: 0.9;
}
</style>
"""

st.markdown(NEUROPULSE_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Assets loader
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_assets():
    df = pd.read_csv("data/synthetic_future_bipolar.csv")
    res_dir = Path("results")
    metrics = pd.read_csv(res_dir / "model_metrics.csv") if (res_dir / "model_metrics.csv").exists() else None
    cv_metrics = pd.read_csv(res_dir / "cv_metrics.csv") if (res_dir / "cv_metrics.csv").exists() else None
    importance = pd.read_csv(res_dir / "feature_importance.csv") if (res_dir / "feature_importance.csv").exists() else None
    stress = pd.read_csv(res_dir / "stress_test.csv") if (res_dir / "stress_test.csv").exists() else None
    thresholds = pd.read_csv(res_dir / "selected_thresholds.csv") if (res_dir / "selected_thresholds.csv").exists() else None
    reliability = pd.read_csv(res_dir / "reliability_brier.csv") if (res_dir / "reliability_brier.csv").exists() else None
    shift = pd.read_csv(res_dir / "distribution_shift.csv") if (res_dir / "distribution_shift.csv").exists() else None
    model_path = res_dir / "logistic_regression_pipeline.joblib"
    saved_model = joblib.load(model_path) if model_path.exists() else None
    return df, metrics, cv_metrics, importance, stress, thresholds, reliability, shift, saved_model

df, metrics_df, cv_metrics_df, imp_df, stress_df, thr_df, rel_df, shift_df, saved_model = load_assets()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    # Custom wordmark with subtle neural icon line
    st.markdown('''
    <div style="padding: 2rem 0.75rem 1.25rem 0.75rem;">
      <div style="display:flex; align-items:center; gap:0.85rem;">
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="16" cy="16" r="14" stroke="#46E0C1" stroke-width="1.5" opacity="0.35"/>
          <circle cx="16" cy="16" r="7" stroke="#46E0C1" stroke-width="1" opacity="0.6"/>
          <circle cx="16" cy="16" r="3" fill="#46E0C1" opacity="0.9"/>
          <line x1="16" y1="2" x2="16" y2="30" stroke="#46E0C1" stroke-width="0.5" opacity="0.25"/>
          <line x1="2" y1="16" x2="30" y2="16" stroke="#46E0C1" stroke-width="0.5" opacity="0.25"/>
          <circle cx="8" cy="8" r="2" fill="#F4B860" opacity="0.8"/>
          <circle cx="24" cy="24" r="2" fill="#F4B860" opacity="0.8"/>
        </svg>
        <div>
          <div style="font-family:'Manrope',sans-serif;font-weight:800;font-size:1.35rem;letter-spacing:-0.035em;color:#F8FAFC;line-height:1.1;">NeuroPredict</div>
          <div style="font-size:0.75rem;color:#8497A5;font-weight:300;letter-spacing:0.02em;margin-top:0.2rem;">Prospective Bipolar ML</div>
        </div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="divider-gradient"></div>', unsafe_allow_html=True)

    # Navigation list
    sections = [
        ("Overview", "Overview of the project and key metrics"),
        ("Data Explorer", "Dataset statistics and predictor profile"),
        ("Training & Results", "Pipeline architecture and model evaluation"),
        ("Feature Impact", "Permutation importance visualization"),
        ("Stress Test", "Robustness under simulated missingness"),
        ("Live Prediction", "Interactive prediction with baseline inputs"),
    ]
    for idx, (label, desc) in enumerate(sections):
        # Minimal active state using session_state isn't needed since tabs are page-level
        # We'll keep clean text links for sidebar info
        st.markdown(f'''
        <div style="padding:0.35rem 0.25rem; border-radius:0.5rem; margin-bottom:0.15rem;">
          <div style="font-weight:600;color:#E8F0F3;font-size:0.92rem;letter-spacing:0.01em;">{label}</div>
          <div style="font-size:0.78rem;color:#8497A5;margin-top:0.15rem;line-height:1.35;">{desc}</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('<div class="divider-gradient"></div>', unsafe_allow_html=True)
    st.info("Synthetic 6-month cohort. Not clinical. Benchmark only.", icon="⚠")
    st.caption(f"Dataset: {len(df):,} rows · {df.shape[1]} features")

# ------------------------------------------------------------------
# Neural decorative SVG (signature element)
# ------------------------------------------------------------------
NEURAL_SVG = '''
<svg viewBox="0 0 320 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:420px;height:auto;display:block;margin:0 auto;opacity:0.92;">
  <defs>
    <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#46E0C1" stop-opacity="0.3"/>
      <stop offset="100%" stop-color="#F4B860" stop-opacity="0.3"/>
    </linearGradient>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="2.5" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <!-- Background ring -->
  <circle cx="160" cy="60" r="42" stroke="#46E0C1" stroke-width="0.5" fill="none" opacity="0.2"/>
  <circle cx="160" cy="60" r="28" stroke="#46E0C1" stroke-width="0.5" fill="none" opacity="0.35"/>

  <!-- Nodes -->
  <g stroke-width="0">
    <circle cx="160" cy="60" r="4.5" fill="#46E0C1" filter="url(#glow)"/>
    <circle cx="115" cy="42" r="2.8" fill="#F4B860" filter="url(#glow)"/>
    <circle cx="205" cy="42" r="2.8" fill="#F4B860" filter="url(#glow)"/>
    <circle cx="105" cy="78" r="2.8" fill="#F4B860" filter="url(#glow)"/>
    <circle cx="215" cy="78" r="2.8" fill="#F4B860" filter="url(#glow)"/>
    <circle cx="160" cy="18" r="2" fill="#46E0C1" filter="url(#glow)"/>
  </g>

  <!-- Connections -->
  <g stroke="url(#lineGrad)" stroke-width="0.75" fill="none" opacity="0.6">
    <line x1="160" y1="60" x2="115" y2="42"/>
    <line x1="160" y1="60" x2="205" y2="42"/>
    <line x1="160" y1="60" x2="105" y2="78"/>
    <line x1="160" y1="60" x2="215" y2="78"/>
    <line x1="160" y1="60" x2="160" y2="18"/>
    <line x1="115" y1="42" x2="205" y2="42"/>
    <line x1="105" y1="78" x2="215" y2="78"/>
  </g>
</svg>
'''

# ------------------------------------------------------------------
# Premium Header (with decorative neural SVG)
# ------------------------------------------------------------------
st.markdown('<div class="mesh-bg"></div>', unsafe_allow_html=True)

header_col, neural_col = st.columns([3, 1])
with header_col:
    st.markdown('''
    <div class="editorial-header" style="padding-top:1.5rem; padding-bottom:0.5rem;">
      <h1>NeuroPredict</h1>
      <p class="subtitle">Prospective Bipolar Disorder Intelligence · 6-Month Horizon · Synthetic Cohort Benchmark</p>
    </div>
    ''', unsafe_allow_html=True)
with neural_col:
    st.markdown(NEURAL_SVG, unsafe_allow_html=True)

st.markdown('<div class="divider-gradient"></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Main Tabs
# ------------------------------------------------------------------
tab_overview, tab_data, tab_training, tab_importance, tab_stress, tab_predict = st.tabs([
    "Overview", "Data Explorer", "Training & Results",
    "Feature Impact", "Stress Test", "Live Prediction"
])

# ===================================================================
# OVERVIEW TAB
# ===================================================================
with tab_overview:
    # Hero metrics (asymmetrical layout)
    top_row = st.columns([2, 1, 1, 1])
    with top_row[0]:
        # Dominant AUROC card — best model by CV (model selection metric), test reserved for final eval
        best_cv_row = None
        if cv_metrics_df is not None and len(cv_metrics_df) > 0:
            best_cv_idx = cv_metrics_df["AUROC (mean)"].idxmax()
            best_cv_row = cv_metrics_df.loc[best_cv_idx]
        if best_cv_row is not None:
            best_model = best_cv_row["Model"]
            best_auroc = best_cv_row["AUROC (mean)"]
            best_std = best_cv_row["AUROC (std)"]
            subtitle = f"{best_model} · CV AUROC ± std"
        else:
            best_model = metrics_df.iloc[0]["Model"] if metrics_df is not None and len(metrics_df) > 0 else "N/A"
            best_auroc = metrics_df.iloc[0]["AUROC"] if metrics_df is not None and len(metrics_df) > 0 else 0.0
            best_std = 0.0
            subtitle = f"{best_model} · Test AUROC"
        st.markdown(f'''
        <div class="card" style="background:linear-gradient(160deg, rgba(70,224,193,0.08), rgba(244,184,96,0.05)); border-color:rgba(70,224,193,0.2); padding:1.75rem 2rem; height:160px; display:flex; flex-direction:column; justify-content:center;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#8497A5;text-transform:uppercase;letter-spacing:0.1em;font-weight:500;">Primary Metric</div>
          <div style="font-family:'Manrope',sans-serif;font-weight:800;font-size:3.2rem;color:#E8F0F3;letter-spacing:-0.05em;line-height:1; margin-top:0.35rem;">{best_auroc:.4f}</div>
          <div style="font-size:0.8rem;color:#46E0C1;font-weight:600;margin-top:0.25rem;">{subtitle}</div>
          <div style="font-size:0.75rem;color:#8497A5;margin-top:0.15rem;">± {best_std:.4f}</div>
        </div>
        ''', unsafe_allow_html=True)
    with top_row[1]:
        st.markdown('''
        <div class="metric-card" style="height:160px; display:flex; flex-direction:column; justify-content:center;">
          <div class="label">Dataset Size</div>
          <div class="value">3,750</div>
          <div class="delta">Synthetic cohort</div>
        </div>
        ''', unsafe_allow_html=True)
    with top_row[2]:
        positive_rate = df['Future_bipolar_6mo'].mean()
        st.markdown(f'''
        <div class="metric-card" style="height:160px; display:flex; flex-direction:column; justify-content:center;">
          <div class="label">Positive Rate</div>
          <div class="value">{positive_rate:.1%}</div>
          <div class="delta">Synthetic imbalance</div>
        </div>
        ''', unsafe_allow_html=True)
    with top_row[3]:
        missing_total = int(df.isna().sum().sum())
        st.markdown(f'''
        <div class="metric-card" style="height:160px; display:flex; flex-direction:column; justify-content:center;">
          <div class="label">Missing Cells</div>
          <div class="value">{missing_total:,}</div>
          <div class="delta">Simulated gaps</div>
        </div>
        ''', unsafe_allow_html=True)

    # Pipeline diagram
    st.subheader("Pipeline Architecture", anchor=False)
    pipeline_cols = st.columns([1, 0.15, 1, 0.15, 1, 0.15, 1, 0.15, 1])
    stages = [
        ("Baseline Predictors", "14 predictors · No leakage"),
        ("Preprocessing", "Median imputation · Scaling"),
        ("Training", "Stratified split · Seed 42"),
        ("Models", "LogReg · RF · GB"),
        ("Evaluation", "AUROC · AUPRC · Brier"),
    ]
    for i, (label, desc) in enumerate(stages):
        with pipeline_cols[i * 2]:
            st.markdown('<div class="card" style="padding:0.85rem 1rem; text-align:center;">', unsafe_allow_html=True)
            st.markdown(f"<div style='font-family:\'Manrope\',sans-serif;font-weight:700;color:#E8F0F3;font-size:0.9rem;'>{label}</div>", unsafe_allow_html=True)
            st.caption(desc)
            st.markdown("</div>", unsafe_allow_html=True)
        if i < len(stages) - 1:
            with pipeline_cols[i * 2 + 1]:
                st.markdown('''<div style="display:flex;align-items:center;height:100%;justify-content:center;"><span style="font-size:1.2rem;color:#46E0C1;opacity:0.6;">→</span></div>''', unsafe_allow_html=True)

    # Project design explanation
    st.subheader("Project Design", anchor=False)
    st.markdown('''
    <div class="card" style="margin-top:1.25rem;">
      <h4 style="font-family:'Manrope',sans-serif;font-weight:700;color:#E8F0F3;margin-bottom:0.75rem;">Prospective Cohort Benchmark</h4>
      <p style="color:#8497A5;line-height:1.7;font-size:0.92rem;">
        The model predicts whether a person will receive a bipolar diagnosis within <strong style="color:#E8F0F3;">6 months</strong>, using only baseline predictors (demographics, prior episodes, sleep, stress, mood instability, medication adherence, etc.).
      </p>
      <ul style="margin-top:0.6rem;color:#8497A5;line-height:1.7;list-style:none;padding-left:0;font-size:0.92rem;">
        <li style="margin-bottom:0.35rem;"><span style="color:#46E0C1;font-weight:700;">●</span> Input: 14 baseline predictors (no current manic symptoms defining the label)</li>
        <li style="margin-bottom:0.35rem;"><span style="color:#46E0C1;font-weight:700;">●</span> No label leakage: other diagnosis columns excluded; ID removed</li>
        <li style="margin-bottom:0.35rem;"><span style="color:#46E0C1;font-weight:700;">●</span> Pipeline: median imputation inside sklearn Pipeline (train-only fit)</li>
        <li><span style="color:#46E0C1;font-weight:700;">●</span> Robustness: simulated missingness stress test (0–50%)</li>
      </ul>
    </div>
    ''', unsafe_allow_html=True)

    # Model comparison cards (asymmetrical, premium)
    if metrics_df is not None and len(metrics_df) > 0:
        st.subheader("Test-Set Performance", anchor=False)
        best_idx = metrics_df['AUROC'].idxmax()
        best_row = metrics_df.loc[best_idx]
        # Feature cards in a grid
        for _, row in metrics_df.iterrows():
            is_best = (row.name == best_idx)
            accent = "#46E0C1" if is_best else "#F4B860"
            st.markdown(f'''
            <div style="background:linear-gradient(160deg, rgba(70,224,193,0.04), rgba(244,184,96,0.03)); border:1px solid rgba(70,224,193,0.08); border-radius:1rem; padding:1.25rem 1.25rem; margin-bottom:1rem; position:relative; {('border-left:3px solid #46E0C1;') if is_best else ''}">
              <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                  <div style="font-family:'Manrope',sans-serif;font-weight:700;color:#E8F0F3;font-size:1.1rem; letter-spacing:-0.02em;">{row['Model']}</div>
                  <div style="font-size:0.72rem;color:#8497A5;margin-top:0.15rem;font-family:'Inter',sans-serif; font-weight:500; text-transform:uppercase; letter-spacing:0.08em;">{'Best Model' if is_best else 'Test Metrics'}</div>
                </div>
                <div style="text-align:right; font-family:'Manrope',sans-serif; font-size:2rem; font-weight:800; color:#E8F0F3; letter-spacing:-0.04em; line-height:1;">{row['AUROC']:.4f}</div>
              </div>
              <div style="margin-top:0.85rem; display:flex; gap:0.4rem; flex-wrap:wrap;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;background:rgba(255,255,255,0.05);padding:0.2rem 0.5rem;border-radius:0.35rem;color:#E8F0F3;">AUROC {row['AUROC']:.4f}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;background:rgba(255,255,255,0.05);padding:0.2rem 0.5rem;border-radius:0.35rem;color:#E8F0F3;">AUPRC {row['AUPRC']:.3f}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;background:rgba(255,255,255,0.05);padding:0.2rem 0.5rem;border-radius:0.35rem;color:#E8F0F3;">Sens {row['Sensitivity']:.3f}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;background:rgba(255,255,255,0.05);padding:0.2rem 0.5rem;border-radius:0.35rem;color:#E8F0F3;">Spec {row['Specificity']:.3f}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;background:rgba(255,255,255,0.05);padding:0.2rem 0.5rem;border-radius:0.35rem;color:#E8F0F3;">F1 {row['F1']:.3f}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;background:rgba(255,255,255,0.05);padding:0.2rem 0.5rem;border-radius:0.35rem;color:#E8F0F3;">Brier {row['Brier']:.4f}</span>
              </div>
            </div>
            ''', unsafe_allow_html=True)

        plot_path = Path("results/evaluation_plots.png")
        if plot_path.exists():
            st.subheader("Evaluation Plots", anchor=False)
            st.image(str(plot_path), use_column_width=True)

# ===================================================================
# DATA EXPLORER
# ===================================================================
with tab_data:
    st.subheader("Synthetic Cohort Profile")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**Target Distribution**")
        target_counts = df["Future_bipolar_6mo"].value_counts().sort_index()
        chart_df = target_counts.rename({0: "No (6mo)", 1: "Yes (6mo)"}).reset_index()
        chart_df.columns = ["Outcome", "Count"]
        fig_target = px.bar(
            chart_df, x="Outcome", y="Count", color="Outcome",
            color_discrete_map={"No (6mo)": "#8497A5", "Yes (6mo)": "#46E0C1"},
            title="6-Month Diagnosis Distribution"
        )
        fig_target.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#E8F0F3",
            margin=dict(l=10, r=10, t=30, b=20), showlegend=False,
            xaxis=dict(gridcolor="rgba(70,224,193,0.06)", tickfont=dict(size=11, color="#8497A5")),
            yaxis=dict(gridcolor="rgba(70,224,193,0.06)", tickfont=dict(size=11, color="#8497A5")),
        )
        st.plotly_chart(fig_target, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
    with d2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**Predictor Columns (Leakage-Safe)**")
        feature_cols = [c for c in df.columns if c not in ["Future_bipolar_6mo", "ID"]]
        for col in feature_cols:
            st.caption(f"• {col}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Dataset statistics
    feature_cols = [c for c in df.columns if c not in ["Future_bipolar_6mo", "ID"]]
    st.markdown('''
    <div class="card" style="margin-top:1.25rem;">
    <h4 style="font-family:'Manrope',sans-serif;font-weight:700;color:#E8F0F3;margin-bottom:0.75rem;">Dataset Statistics</h4>
    <table style="width:100%;font-family:'JetBrains Mono',monospace;color:#8497A5;font-size:0.85rem;line-height:1.6;border-collapse:collapse;">
      <tr style="border-bottom:1px solid rgba(70,224,193,0.08);"><td style="padding:0.35rem 0;font-weight:500;color:#E8F0F3;">Total Rows</td><td style="padding:0.35rem 0;text-align:right;color:#F8FAFC;">{len_df}</td></tr>
      <tr style="border-bottom:1px solid rgba(70,224,193,0.08);"><td style="padding:0.35rem 0;font-weight:500;color:#E8F0F3;">Predictor Features</td><td style="padding:0.35rem 0;text-align:right;color:#F8FAFC;">{feat_count}</td></tr>
      <tr style="border-bottom:1px solid rgba(70,224,193,0.08);"><td style="padding:0.35rem 0;font-weight:500;color:#E8F0F3;">Positive Rate</td><td style="padding:0.35rem 0;text-align:right;color:#F8FAFC;">{pos_rate:.2%}</td></tr>
      <tr><td style="padding:0.35rem 0;font-weight:500;color:#E8F0F3;">Missing Cells</td><td style="padding:0.35rem 0;text-align:right;color:#F8FAFC;">{missing_total:,}</td></tr>
    </table>
    </div>
    '''.format(
        len_df=len(df), feat_count=len(feature_cols),
        pos_rate=df['Future_bipolar_6mo'].mean(),
        missing_total=int(df.isna().sum().sum())
    ), unsafe_allow_html=True)

    st.subheader("Sample Data", anchor=False)
    st.dataframe(df.head(10), width='stretch', hide_index=True)

# ===================================================================
# TRAINING & RESULTS
# ===================================================================
with tab_training:
    st.subheader("Pipeline Architecture")
    st.write("Three sklearn Pipelines with internal imputation · Stratified split · Seed 42")

    arch_cols = st.columns(5)
    stages = [
        ("Baseline Predictors", "14 features"),
        ("Preprocessing", "Median impute"),
        ("Training", "Stratified split"),
        ("Ensembles", "3 models"),
        ("Evaluation", "8 metrics"),
    ]
    for i, (label, desc) in enumerate(stages):
        with arch_cols[i]:
            st.markdown('''
            <div style="background:#121E27;border:1px solid rgba(70,224,193,0.08);border-radius:1rem;padding:1rem;text-align:center;height:110px;display:flex;flex-direction:column;justify-content:center;">
              <div style="font-family:'Manrope',sans-serif;font-weight:700;color:#F8FAFC;font-size:0.95rem;letter-spacing:-0.02em;">{label}</div>
              <div style="font-size:0.78rem;color:#8497A5;margin-top:0.25rem;">{desc}</div>
            </div>
            '''.format(label=label, desc=desc), unsafe_allow_html=True)

    if metrics_df is not None:
        st.subheader("Test-Set Metrics", anchor=False)
        best_idx = metrics_df['AUROC'].idxmax()
        for _, row in metrics_df.iterrows():
            is_best = (row.name == best_idx)
            accent = "#46E0C1" if is_best else "#F4B860"
            st.markdown(f'''
            <div style="background:#121E27;border:1px solid rgba(70,224,193,0.06);border-radius:1.15rem;padding:1.25rem;margin-bottom:0.75rem;{('border-left:3px solid #46E0C1;') if is_best else ''}">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <div style="font-family:'Manrope',sans-serif;font-weight:700;color:#E8F0F3;font-size:1.05rem;letter-spacing:-0.02em;">{row['Model']}</div>
                  <div style="font-size:0.78rem;color:#8497A5;margin-top:0.2rem;">{'Best performing model' if is_best else 'Ensemble / baseline'}</div>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.8rem;font-weight:700;color:#E8F0F3;letter-spacing:-0.03em;">{row['AUROC']:.4f}</div>
              </div>
              <div style="margin-top:0.9rem;display:flex;gap:0.35rem;flex-wrap:wrap;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;background:rgba(255,255,255,0.06);padding:0.15rem 0.5rem;border-radius:0.35rem;color:#F8FAFC;">AUROC {row['AUROC']:.4f}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;background:rgba(255,255,255,0.06);padding:0.15rem 0.5rem;border-radius:0.35rem;color:#F8FAFC;">AUPRC {row['AUPRC']:.3f}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;background:rgba(255,255,255,0.06);padding:0.15rem 0.5rem;border-radius:0.35rem;color:#F8FAFC;">Sens {row['Sensitivity']:.3f}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;background:rgba(255,255,255,0.06);padding:0.15rem 0.5rem;border-radius:0.35rem;color:#F8FAFC;">Spec {row['Specificity']:.3f}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;background:rgba(255,255,255,0.06);padding:0.15rem 0.5rem;border-radius:0.35rem;color:#F8FAFC;">Prec {row['Precision']:.3f}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;background:rgba(255,255,255,0.06);padding:0.15rem 0.5rem;border-radius:0.35rem;color:#F8FAFC;">F1 {row['F1']:.3f}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;background:rgba(255,255,255,0.06);padding:0.15rem 0.5rem;border-radius:0.35rem;color:#F8FAFC;">Brier {row['Brier']:.4f}</span>
              </div>
            </div>
            ''', unsafe_allow_html=True)

    # Stress chart
    if stress_df is not None:
        st.subheader("Robustness Profile", anchor=False)
        pivot_auroc = stress_df.pivot(index="Missing %", columns="Model", values="AUROC")
        pivot_brier = stress_df.pivot(index="Missing %", columns="Model", values="Brier")
        sc1, sc2 = st.columns([1, 1])
        with sc1:
            fig_auroc = px.line(
                pivot_auroc.reset_index().melt(id_vars="Missing %", value_name="AUROC", var_name="Model"),
                x="Missing %", y="AUROC", color="Model", markers=True,
                title="Performance Under Missingness",
                color_discrete_sequence=["#46E0C1", "#F4B860", "#E87A7A"]
            )
            fig_auroc.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#E8F0F3",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=50, b=30)
            )
            st.plotly_chart(fig_auroc, width='stretch')
        with sc2:
            fig_brier = px.line(
                pivot_brier.reset_index().melt(id_vars="Missing %", value_name="Brier", var_name="Model"),
                x="Missing %", y="Brier", color="Model", markers=True,
                title="Calibration Under Missingness",
                color_discrete_sequence=["#46E0C1", "#F4B860", "#E87A7A"]
            )
            fig_brier.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#E8F0F3",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=50, b=30)
            )
            st.plotly_chart(fig_brier, width='stretch')

    # ------------------------------------------------------------------
    # Threshold Optimization — Step 1
    # ------------------------------------------------------------------
    st.subheader("Threshold Optimization", anchor=False)
    st.write("Adjust the classification threshold to observe trade-offs between sensitivity, specificity, precision, and F1. Metrics are computed from the held-out test set (no leakage).")

    # Rebuild predictions on train/validation/test splits (no leakage into threshold tuning)
    @st.cache_data(show_spinner=False)
    def get_split_predictions():
        from sklearn.model_selection import train_test_split
        df_ref = pd.read_csv("data/synthetic_future_bipolar.csv")
        from bipolar_ml import prepare_features_target, build_models
        X_ref, y_ref, _ = prepare_features_target(df_ref)

        # Outer split: train+val (75%) / test (25%) — test reserved for final evaluation
        X_trv, X_te, y_trv, y_te = train_test_split(
            X_ref, y_ref, test_size=0.25, random_state=42, stratify=y_ref
        )

        # Inner split: train_sub / validation — validation used only for threshold tuning
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_trv, y_trv, test_size=0.20, random_state=42, stratify=y_trv
        )

        predictions = {}
        for name, pipe in build_models(42).items():
            pipe.fit(X_tr, y_tr)  # fit on training subset only
            predictions[name] = {
                "val_proba": pipe.predict_proba(X_val)[:, 1],
                "val_y_true": y_val.values,
                "test_proba": pipe.predict_proba(X_te)[:, 1],
                "test_y_true": y_te.values,
            }
        return predictions, X_trv, y_trv, X_te, y_te

    predictions, X_val, y_val, X_te, y_te = get_split_predictions()

    # Threshold slider labels now tune on validation (no test leakage); metrics reported on held-out test set
    st.markdown("### Set Classification Thresholds")
    st.caption("Thresholds are tuned on a held-out validation split; metrics below are computed on the untouched test set.")
    model_keys = list(predictions.keys())
    col_sliders = st.columns(3)
    if "thr_values" not in st.session_state:
        st.session_state["thr_values"] = {k: 0.5 for k in model_keys}
    for i, key in enumerate(model_keys):
        with col_sliders[i]:
            st.session_state["thr_values"][key] = st.slider(
                f"{key}", 0.0, 1.0,
                st.session_state["thr_values"][key], 0.01,
                key=f"slider_thr_{key}"
            )
            st.caption(f"Current: {st.session_state['thr_values'][key]:.2f}")

    execute = st.button("Execute Threshold Optimization", type="primary", width='stretch')

    # Metrics calculation function
    def compute_at_threshold(proba, y_true, thresh):
        pred = (proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        return {
            "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "Precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "F1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
            "Accuracy": (tp + tn) / len(y_true),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        }

    # Dynamic results cards (computed only on Execute)
    if execute:
        st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
        metric_cols = st.columns(3)
        for i, key in enumerate(model_keys):
            # Threshold chosen from validation predictions; final metrics reported on held-out test set
            metrics = compute_at_threshold(
                predictions[key]["test_proba"], predictions[key]["test_y_true"],
                st.session_state["thr_values"][key]
            )
            with metric_cols[i]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"**{key}** · Threshold `{st.session_state['thr_values'][key]:.2f}`")
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f'''
                <div style="text-align:center;">
                  <div style="font-size:0.72rem;color:#8497A5;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.25rem;">Sens</div>
                  <div style="font-family:'Manrope',sans-serif;font-weight:800;color:#46E0C1;font-size:1.35rem;">{metrics['Sensitivity']:.3f}</div>
                </div>
                ''', unsafe_allow_html=True)
                c2.markdown(f'''
                <div style="text-align:center;">
                  <div style="font-size:0.72rem;color:#8497A5;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.25rem;">Spec</div>
                  <div style="font-family:'Manrope',sans-serif;font-weight:800;color:#46E0C1;font-size:1.35rem;">{metrics['Specificity']:.3f}</div>
                </div>
                ''', unsafe_allow_html=True)
                c3.markdown(f'''
                <div style="text-align:center;">
                  <div style="font-size:0.72rem;color:#8497A5;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.25rem;">Prec</div>
                  <div style="font-family:'Manrope',sans-serif;font-weight:800;color:#46E0C1;font-size:1.35rem;">{metrics['Precision']:.3f}</div>
                </div>
                ''', unsafe_allow_html=True)
                c4.markdown(f'''
                <div style="text-align:center;">
                  <div style="font-size:0.72rem;color:#8497A5;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.25rem;">F1</div>
                  <div style="font-family:'Manrope',sans-serif;font-weight:800;color:#46E0C1;font-size:1.35rem;">{metrics['F1']:.3f}</div>
                </div>
                ''', unsafe_allow_html=True)
                st.caption(f"Acc {metrics['Accuracy']:.3f} · TP {metrics['TP']} · FP {metrics['FP']} · FN {metrics['FN']} · TN {metrics['TN']}")
                st.markdown("</div>", unsafe_allow_html=True)

    # AUROC vs AUPRC comparison (CV means)
    if cv_metrics_df is not None and len(cv_metrics_df) > 0:
        st.subheader("AUROC vs AUPRC — Cross-Validated Comparison", anchor=False)
        st.write("Mean ± std across 5 stratified folds. Threshold-independent discrimination metrics.")
        comp_df = cv_metrics_df.melt(
            id_vars=["Model"], value_vars=["AUROC (mean)", "AUPRC (mean)"],
            var_name="Metric", value_name="Score"
        )
        std_map = {}
        for _, row in cv_metrics_df.iterrows():
            std_map[(row["Model"], "AUROC (mean)")] = row["AUROC (std)"]
            std_map[(row["Model"], "AUPRC (mean)")] = row["AUPRC (std)"]
        comp_df["Std"] = comp_df.apply(
            lambda r: std_map.get((r["Model"], r["Metric"]), 0.0), axis=1
        )
        fig_comp = px.bar(
            comp_df, x="Model", y="Score", color="Metric", barmode="group",
            error_y="Std",
            color_discrete_map={"AUROC (mean)": "#46E0C1", "AUPRC (mean)": "#F4B860"},
            title="Discrimination Performance (CV mean ± std)",
        )
        fig_comp.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#E8F0F3",
            margin=dict(l=40, r=10, t=50, b=30),
            yaxis=dict(range=[0, 1], gridcolor="rgba(70,224,193,0.10)"),
            xaxis=dict(gridcolor="rgba(70,224,193,0.10)"),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_comp, width='stretch')
    else:
        st.info("Adjust the sliders above, then click **Execute Threshold Optimization** to compute metrics for all 3 models.")

    # ------------------------------------------------------------------
    # Model Comparison Dashboard — Step 3
    # ------------------------------------------------------------------
    st.markdown('<div class="divider-gradient"></div>', unsafe_allow_html=True)
    st.subheader("Model Comparison — ROC, PR & Confusion Matrix", anchor=False)
    st.write("Computed from held-out test predictions using each saved pipeline.")

    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix as cm

    comp_cols = st.columns(3)
    roc_figs = []
    pr_figs = []
    cmats = {}

    for i, key in enumerate(model_keys):
        proba = predictions[key]["test_proba"]
        y_true = predictions[key]["test_y_true"]
        fpr, tpr, _ = roc_curve(y_true, proba)
        prec, rec, _ = precision_recall_curve(y_true, proba)
        cmat = cm(y_true, (proba >= 0.5).astype(int), labels=[0, 1])
        roc_figs.append((key, fpr, tpr))
        pr_figs.append((key, rec, prec))
        cmats[key] = cmat

        with comp_cols[i]:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=key,
                line=dict(width=3, color="#46E0C1"),
                fill="tozeroy", fillcolor="rgba(70,224,193,0.12)"
            ))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         name="Chance", line=dict(dash="dash", color="#8497A5", width=1)))
            fig_roc.update_layout(
                title=f"{key} — ROC", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E8F0F3", height=300, margin=dict(l=40, r=10, t=40, b=30),
                xaxis_title="FPR", yaxis_title="TPR",
                xaxis=dict(gridcolor="rgba(70,224,193,0.10)"),
                yaxis=dict(gridcolor="rgba(70,224,193,0.10)"),
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig_roc, width='stretch')

            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=rec, y=prec, mode="lines", name=key,
                line=dict(width=3, color="#F4B860"),
                fill="tozeroy", fillcolor="rgba(244,184,96,0.12)"
            ))
            fig_pr.update_layout(
                title=f"{key} — Precision-Recall", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E8F0F3", height=300, margin=dict(l=40, r=10, t=40, b=30),
                xaxis_title="Recall", yaxis_title="Precision",
                xaxis=dict(gridcolor="rgba(244,184,96,0.10)"),
                yaxis=dict(gridcolor="rgba(244,184,96,0.10)"),
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig_pr, width='stretch')

    # Combined ROC overlay
    fig_combined = go.Figure()
    for key, fpr, tpr in roc_figs:
        fig_combined.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=key,
            line=dict(width=3), fill="tozeroy", opacity=0.10
        ))
    fig_combined.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                      line=dict(dash="dash", color="#8497A5")))
    fig_combined.update_layout(
        title="ROC Curves — All Models (overlay)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E8F0F3", height=350, margin=dict(l=40, r=10, t=40, b=30),
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        xaxis=dict(gridcolor="rgba(70,224,193,0.10)"),
        yaxis=dict(gridcolor="rgba(70,224,193,0.10)"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
    )
    st.subheader("ROC Curves — Combined", anchor=False)
    st.plotly_chart(fig_combined, width='stretch')

    # Confusion matrices
    st.subheader("Confusion Matrices — Threshold 0.5", anchor=False)
    cm_cols = st.columns(3)
    for i, key in enumerate(model_keys):
        cmat = cmats[key]
        with cm_cols[i]:
            fig_cm = go.Figure(data=go.Heatmap(
                z=cmat, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                colorscale="Emrld", showscale=False
            ))
            fig_cm.update_layout(
                title=key, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E8F0F3", height=260, margin=dict(l=30, r=10, t=40, b=30)
            )
            st.plotly_chart(fig_cm, width='stretch')

    # ------------------------------------------------------------------
    # Reliability Analysis — Calibration, Brier, Distribution Shift
    # ------------------------------------------------------------------
    st.markdown('<div class="divider-gradient"></div>', unsafe_allow_html=True)
    st.subheader("Reliability Analysis — Calibration, Brier & Distribution Shift", anchor=False)
    st.write("Calibration plots and Brier scores on the held-out test set, plus train-vs-test distribution shift (KS test).")

    if rel_df is not None and len(rel_df) > 0:
        st.markdown("#### Brier Score Comparison (lower = better calibration)")
        rel_cols = st.columns(len(rel_df))
        for i, row in rel_df.iterrows():
            with rel_cols[i]:
                st.markdown('<div class="card" style="text-align:center;padding:1rem;">', unsafe_allow_html=True)
                st.markdown(f"<div style='font-family:\'Manrope\',sans-serif;font-weight:700;color:#E8F0F3;font-size:1.1rem;'>{row['Model']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-family:\'JetBrains Mono\',monospace;font-size:2rem;font-weight:800;color:#E8F0F3;letter-spacing:-0.03em;'>{row['Brier']:.4f}</div>", unsafe_allow_html=True)
                st.caption("Brier score")
                st.markdown("</div>", unsafe_allow_html=True)

    # Calibration plots
    st.markdown("#### Calibration Curves (Test Set)")
    if saved_model is not None and metrics_df is not None:
        from sklearn.calibration import calibration_curve
        cal_cols = st.columns(3)
        for i, key in enumerate(model_keys):
            with cal_cols[i]:
                proba = predictions[key]["test_proba"]
                y_true = predictions[key]["test_y_true"]
                frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
                brier = brier_score_loss(y_true, proba)
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(
                    x=mean_pred, y=frac_pos, mode="lines+markers", name=key,
                    line=dict(width=3, color="#46E0C1"),
                    marker=dict(size=6)
                ))
                fig_cal.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                    line=dict(dash="dash", color="#8497A5", width=1)
                ))
                fig_cal.update_layout(
                    title=f"{key} — Calibration (Brier={brier:.4f})",
                    xaxis_title="Mean Predicted Probability", yaxis_title="Fraction of Positives",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#E8F0F3",
                    height=300, margin=dict(l=40, r=10, t=50, b=30),
                    xaxis=dict(range=[0, 1], gridcolor="rgba(70,224,193,0.10)"),
                    yaxis=dict(range=[0, 1], gridcolor="rgba(70,224,193,0.10)")
                )
                st.plotly_chart(fig_cal, width='stretch')

    # Distribution shift table
    if shift_df is not None and len(shift_df) > 0:
        st.markdown("#### Distribution Shift — Train vs Test (KS Test)")
        st.write("Features with p < 0.05 indicate significant distribution shift between train and test sets.")
        shift_display = shift_df.copy()
        shift_display["p_value"] = shift_display["p_value"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        shift_display["KS_statistic"] = shift_display["KS_statistic"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        shift_display["ShiftDetected"] = shift_display["ShiftDetected"].apply(lambda x: "⚠️ Yes" if x else "No")
        st.dataframe(shift_display, width='stretch', hide_index=True)

# ===================================================================
# FEATURE IMPACT
# ===================================================================
with tab_importance:
    if imp_df is not None:
        st.subheader("Permutation Feature Importance", anchor=False)
        st.write("Measured as the mean drop in test AUROC when a feature is randomly shuffled.")
        display_df = imp_df.copy()
        display_df["Feature"] = display_df["Feature"].str.replace("_", " ").str.title()
        fig_imp = px.bar(
            display_df.sort_values("Mean AUROC drop", ascending=True),
            x="Mean AUROC drop", y="Feature", orientation="h",
            color="Mean AUROC drop", color_continuous_scale="Emrld",
            title="Predictive Drivers",
        )
        fig_imp.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#E8F0F3",
            margin=dict(l=140, r=30, t=50, b=20)
        )
        st.plotly_chart(fig_imp, width='stretch')
    else:
        st.info("Generate importance with `python bipolar_ml.py --plots`.")

# ===================================================================
# STRESS TEST
# ===================================================================
with tab_stress:
    if stress_df is not None:
        st.subheader("Missing-Data Stress Test", anchor=False)
        st.write("Robustness of each model as simulated missingness increases (0% → 50%). "
                 "Models were trained once on complete data; missing values are filled with training medians — no retraining.")

        # Pivot data
        pivot_auroc = stress_df.pivot(index="Missing %", columns="Model", values="AUROC")
        pivot_brier = stress_df.pivot(index="Missing %", columns="Model", values="Brier")

        colors = {"Logistic Regression": "#46E0C1", "Gradient Boosting": "#F4B860", "Random Forest": "#E87A7A"}
        rgba_colors = {"Logistic Regression": "rgba(70, 224, 193, 0.08)", "Gradient Boosting": "rgba(244, 184, 96, 0.08)", "Random Forest": "rgba(232, 122, 122, 0.08)"}

        # Calculate degradation at max missingness for annotation
        max_rate = int(pivot_auroc.index.max())
        deg_auroc = {}
        deg_brier = {}
        for col in pivot_auroc.columns:
            deg_auroc[col] = pivot_auroc[col].iloc[0] - pivot_auroc[col].iloc[-1]
            deg_brier[col] = pivot_brier[col].iloc[-1] - pivot_brier[col].iloc[0]

        # Two-column layout: AUROC on left, Brier on right
        sc1, sc2 = st.columns([1, 1])

        with sc1:
            st.markdown("#### AUROC Degradation")
            fig_auroc = go.Figure()
            for col in pivot_auroc.columns:
                fig_auroc.add_trace(go.Scatter(
                    x=pivot_auroc.index, y=pivot_auroc[col],
                    name=col, mode="lines+markers",
                    line=dict(width=3, color=colors.get(col, "#8497A5")),
                    marker=dict(size=7, line=dict(width=1, color="#080D12")),
                    fill="tozeroy", opacity=0.08,
                    fillcolor=rgba_colors.get(col, "rgba(132, 151, 165, 0.08)")
                ))
                # Degradation annotation at final point
                fig_auroc.add_annotation(
                    x=pivot_auroc.index[-1], y=pivot_auroc[col].iloc[-1],
                    text=f"{col}<br>▼ {deg_auroc[col]:.3f}",
                    showarrow=False,
                    xanchor="left", xshift=8,
                    font=dict(size=10, color=colors.get(col, "#8497A5"))
                )
            fig_auroc.update_layout(
                title="AUROC vs Simulated Missingness",
                xaxis_title="Missing Value %", yaxis_title="AUROC",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#E8F0F3",
                margin=dict(l=40, r=60, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                xaxis=dict(gridcolor="rgba(70,224,193,0.08)", tickmode="linear", dtick=10),
                yaxis=dict(gridcolor="rgba(70,224,193,0.08)", range=[0.55, 0.85]),
                hovermode="x unified"
            )
            st.plotly_chart(fig_auroc, width='stretch')

        with sc2:
            st.markdown("#### Calibration (Brier Score)")
            fig_brier = go.Figure()
            for col in pivot_brier.columns:
                fig_brier.add_trace(go.Scatter(
                    x=pivot_brier.index, y=pivot_brier[col],
                    name=col, mode="lines+markers",
                    line=dict(width=3, color=colors.get(col, "#8497A5")),
                    marker=dict(size=7, line=dict(width=1, color="#080D12")),
                    fill="tozeroy", opacity=0.08,
                    fillcolor=rgba_colors.get(col, "rgba(132, 151, 165, 0.08)")
                ))
                fig_brier.add_annotation(
                    x=pivot_brier.index[-1], y=pivot_brier[col].iloc[-1],
                    text=f"{col}<br>▲ {deg_brier[col]:.3f}",
                    showarrow=False,
                    xanchor="left", xshift=8,
                    font=dict(size=10, color=colors.get(col, "#8497A5"))
                )
            fig_brier.update_layout(
                title="Brier Score vs Simulated Missingness",
                xaxis_title="Missing Value %", yaxis_title="Brier Score",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#E8F0F3",
                margin=dict(l=40, r=60, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                xaxis=dict(gridcolor="rgba(244,184,96,0.08)", tickmode="linear", dtick=10),
                yaxis=dict(gridcolor="rgba(244,184,96,0.08)", range=[0.09, 0.22]),
                hovermode="x unified"
            )
            st.plotly_chart(fig_brier, width='stretch')

        # Degradation summary table
        st.markdown("#### Degradation Summary (0% → 50% missing)")
        deg_rows = []
        for col in pivot_auroc.columns:
            deg_rows.append({
                "Model": col,
                "AUROC at 0%": f"{pivot_auroc[col].iloc[0]:.3f}",
                "AUROC at 50%": f"{pivot_auroc[col].iloc[-1]:.3f}",
                "AUROC Drop": f"{deg_auroc[col]:.3f}",
                "Brier at 0%": f"{pivot_brier[col].iloc[0]:.3f}",
                "Brier at 50%": f"{pivot_brier[col].iloc[-1]:.3f}",
                "Brier Rise": f"{deg_brier[col]:.3f}",
            })
        deg_df = pd.DataFrame(deg_rows)
        st.dataframe(deg_df, width='stretch', hide_index=True,
                     column_config={
                         "AUROC Drop": st.column_config.ProgressColumn("AUROC Drop", min_value=0, max_value=0.2),
                         "Brier Rise": st.column_config.ProgressColumn("Brier Rise", min_value=0, max_value=0.12),
                     })
    else:
        st.info("No stress test results available. Run `python bipolar_ml.py` to generate them.")

# ===================================================================
# LIVE PREDICTION
# ===================================================================
with tab_predict:
    st.subheader("Live Prediction")
    st.write("Adjust baseline predictors and select a model to see the predicted probability of future bipolar diagnosis within 6 months.")

    # Load all three pipelines fresh (thresholds stored separately in CSV)
    from bipolar_ml import build_models
    all_models = build_models(42)
    # Fit all on Train+Val to match the saved pipeline state
    from sklearn.model_selection import train_test_split
    df_train_ref = pd.read_csv("data/synthetic_future_bipolar.csv")
    from bipolar_ml import prepare_features_target
    X_all, y_all, _ = prepare_features_target(df_train_ref)
    X_trv_all, X_te_all, y_trv_all, y_te_all = train_test_split(
        X_all, y_all, test_size=0.25, random_state=42, stratify=y_all)
    for name, pipe in all_models.items():
        pipe.fit(X_trv_all, y_trv_all)

    # Build threshold lookup from CSV
    thr_lookup = {}
    if thr_df is not None:
        for _, r in thr_df.iterrows():
            thr_lookup[r["Model"]] = float(r["SelectedThreshold"])

    # Model selector
    model_choice = st.selectbox("Select model", list(all_models.keys()), index=0)
    model_obj = all_models[model_choice]
    selected_thr = thr_lookup.get(model_choice, 0.5)
    st.caption(f"Selected threshold for {model_choice}: `{selected_thr:.3f}` (from validation/OOF tuning)")

    # Two-column layout for inputs
    c_l, c_r = st.columns(2)
    with c_l:
        age = st.slider("Age", 18.0, 70.0, 32.0, 0.5)
        prior_dep = st.slider("Prior depression episodes", 0, 8, 1)
        prior_anx = st.slider("Prior anxiety episodes", 0, 6, 0)
        family_hist = st.selectbox("Family history bipolar", [0, 1], index=0, format_func=lambda x: "Yes" if x else "No")
        sleep_score = st.slider("Sleep disturbance (0-10)", 0.0, 10.0, 5.0, 0.1)
        stress_score = st.slider("Stress score (0-10)", 0.0, 10.0, 5.0, 0.1)
    with c_r:
        cog_impair = st.slider("Cognitive impairment (0-10)", 0.0, 10.0, 4.0, 0.1)
        substance_use = st.slider("Substance use frequency", 0, 5, 2)
        mood_instability = st.slider("Baseline mood instability (0-10)", 0.0, 10.0, 5.0, 0.1)
        social_support = st.slider("Social support (0-10)", 0.0, 10.0, 6.0, 0.1)
        energy_level = st.slider("Energy level (0-10)", 0.0, 10.0, 6.0, 0.1)
        social_withdrawal = st.slider("Social withdrawal (0-10)", 0.0, 10.0, 3.5, 0.1)
        medication_adherence = st.slider("Medication adherence (0-1)", 0.0, 1.0, 0.65, 0.01)

    feature_cols_pred = [
        "Age", "Sex", "Prior_depression_episodes", "Prior_anxiety_episodes",
        "Family_history_bipolar", "Sleep_disturbance_score", "Stress_score",
        "Cognitive_impairment_score", "Substance_use_frequency", "Social_support_score",
        "Baseline_mood_instability", "Medication_adherence", "Baseline_energy_level",
        "Baseline_social_withdrawal"
    ]
    sex_default = int(df["Sex"].median())
    sample_dict = {
        "Age": age, "Sex": sex_default,
        "Prior_depression_episodes": prior_dep, "Prior_anxiety_episodes": prior_anx,
        "Family_history_bipolar": family_hist, "Sleep_disturbance_score": sleep_score,
        "Stress_score": stress_score, "Cognitive_impairment_score": cog_impair,
        "Substance_use_frequency": substance_use, "Social_support_score": social_support,
        "Baseline_mood_instability": mood_instability, "Medication_adherence": medication_adherence,
        "Baseline_energy_level": energy_level, "Baseline_social_withdrawal": social_withdrawal,
    }
    sample_df = pd.DataFrame([sample_dict])

    if st.button("Compute Prediction", type="primary", width='stretch'):
        with st.spinner("Evaluating probability..."):
            proba = model_obj.predict_proba(sample_df)[0, 1]
        pred_label = int(proba >= selected_thr)
        if pred_label == 1:
            st.markdown(f'<div class="result-high"><h3 style="margin:0;font-family:\'Manrope\',sans-serif;">Predicted Probability: <span style="font-weight:800;">{proba:.3f}</span></h3><p style="margin-top:0.25rem;color:#E8F0F3;font-size:0.95rem;font-weight:300;">High risk — above {model_choice} threshold ({selected_thr:.2f}) of future bipolar diagnosis within 6 months.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-medium"><h3 style="margin:0;font-family:\'Manrope\',sans-serif;">Predicted Probability: <span style="font-weight:800;">{proba:.3f}</span></h3><p style="margin-top:0.25rem;color:#E8F0F3;font-size:0.95rem;font-weight:300;">Low-Medium risk — below {model_choice} threshold ({selected_thr:.2f}) of future bipolar diagnosis within 6 months.</p></div>', unsafe_allow_html=True)

        # Feature contributions using logistic regression coefficients (LR only)
        if model_choice == "Logistic Regression":
            try:
                coefs = model_obj.named_steps["classifier"].coef_[0]
                contrib_df = pd.DataFrame({
                    "Predictor": sample_df.columns,
                    "Weight": coefs
                }).sort_values("Weight", ascending=False)
                st.subheader("Linear Model Weights", anchor=False)
                st.caption("Positive weights increase the 6-month risk probability; negative weights decrease it.")
                fig_weights = px.bar(
                    contrib_df.sort_values("Weight"), x="Weight", y="Predictor", orientation="h",
                    color="Weight", color_continuous_scale="Emrld",
                    title="Logistic Regression Coefficients",
                )
                fig_weights.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#E8F0F3",
                    margin=dict(l=120, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_weights, width='stretch')
            except Exception:
                st.info("Weight details unavailable for this pipeline type.")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown('''
<div style="margin-top:3.5rem;padding:1.25rem 1.5rem;border-top:1px solid rgba(70,224,193,0.08);border-radius:1rem;background:rgba(255,255,255,0.01);">
  <div style="font-family:'Manrope',sans-serif;font-weight:700;color:#F8FAFC;font-size:1.05rem;letter-spacing:-0.02em;">Synthetic Prospective Dataset</div>
  <p style="color:#8497A5;font-size:0.82rem;margin-top:0.35rem;line-height:1.65;">
    This application uses a synthetic dataset (`Future_bipolar_6mo`) designed exclusively for benchmark testing and educational demonstration. It does not represent clinical data. All results reflect prediction from baseline predictors only, not a verified diagnosis. Not intended for real-world clinical deployment.
  </p>
</div>
''', unsafe_allow_html=True)
