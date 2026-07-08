"""
Run with: python -m streamlit run ecg_dashboard.py
Requires:  ecg_data.json, ecg_models.pkl, ecg_scaler.pkl  
           OR upload ecg_enhanced.xlsx 
"""

import json, pickle, warnings, io, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ECG SIGNAL CLASSIFICATION AND VISUALIZATION FOR HEART DISEASE DETECTION ",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg-dark:     #0A0E1A;
    --bg-card:     #111827;
    --bg-card2:    #1A2235;
    --accent-teal: #00D4B8;
    --accent-pink: #FF4B8B;
    --accent-blue: #4B8BFF;
    --accent-warn: #FFB74D;
    --text-main:   #E8EDF5;
    --text-muted:  #7A8BA3;
    --border:      rgba(0,212,184,0.15);
}

/* ── App background ── */
.stApp { background-color: var(--bg-dark); color: var(--text-main); }
.stApp > header { background: transparent !important; }
[data-testid="stSidebar"] { background-color: #0D1220 !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text-main) !important; }

/* ── Typography ── */
h1,h2,h3,h4 { font-family: 'Space Mono', monospace !important; color: var(--accent-teal) !important; }
p, div, span, label { font-family: 'DM Sans', sans-serif !important; color: var(--text-main); }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px !important;
}
[data-testid="metric-container"] label { color: var(--text-muted) !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--accent-teal) !important; font-family: 'Space Mono', monospace !important; font-size: 1.6rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-teal), var(--accent-blue)) !important;
    color: #0A0E1A !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Select boxes & inputs ── */
.stSelectbox > div, .stNumberInput > div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card) !important; border-radius: 10px; gap: 4px; padding: 6px; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--text-muted) !important; border-radius: 8px; font-family: 'Space Mono', monospace !important; font-size: 13px; }
.stTabs [aria-selected="true"] { background: var(--accent-teal) !important; color: #0A0E1A !important; }

/* ── Dataframes ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px; overflow: hidden; }

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}

/* ── Prediction result box ── */
.pred-box {
    background: linear-gradient(135deg, #0D2A38, #0D2019);
    border: 2px solid var(--accent-teal);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 12px 0;
}
.pred-label { font-family: 'Space Mono', monospace; font-size: 3.5rem; font-weight: 700; color: var(--accent-teal); }
.pred-desc  { font-family: 'DM Sans', sans-serif; font-size: 1.1rem; color: var(--text-muted); margin-top: 8px; }
.pred-conf  { font-family: 'Space Mono', monospace; font-size: 1rem; color: var(--accent-pink); margin-top: 6px; }

/* ── Status badge ── */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    margin: 4px 3px;
}
.badge-N { background: #0D3028; color: #00D4B8; border: 1px solid #00D4B8; }
.badge-S { background: #0D1F3C; color: #4B8BFF; border: 1px solid #4B8BFF; }
.badge-V { background: #3C0D1A; color: #FF4B8B; border: 1px solid #FF4B8B; }
.badge-F { background: #3C2A0D; color: #FFB74D; border: 1px solid #FFB74D; }
.badge-U { background: #231A3C; color: #B47BFF; border: 1px solid #B47BFF; }

/* ── Section headers ── */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* ── Sidebar nav ── */
.sidebar-logo {
    font-family: 'Space Mono', monospace;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--accent-teal) !important;
    padding: 6px 0 4px;
}
.sidebar-sub {
    font-size: 0.75rem;
    color: var(--text-muted) !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 24px;
}

/* ── Matplotlib figure frames ── */
[data-testid="stImage"] img,
.stPlotlyChart, canvas { border-radius: 12px; }
            
/* Fix dropdown contrast */
.stSelectbox div[data-baseweb="select"] {
    background-color: #1A2235 !important;   /* box color */
    color: #E8EDF5 !important;              /* text color */
}

.stSelectbox span {
    color: #E8EDF5 !important;              /* selected text */
}

.stSelectbox svg {
    color: #00D4B8 !important;              /* dropdown arrow */
}
            
/* Remove top white bar */
header[data-testid="stHeader"] {
    background-color: #0A0E1A !important;
}

/* Remove extra padding above title */
.block-container {
    padding-top: 1rem !important;
}
            
/* Fix dropdown selected value (inside box) */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span {
    color: #000000 !important;
    opacity: 1 !important;
}

/* Fix dropdown options (the list) */
div[data-baseweb="popover"] ul li,
div[data-baseweb="menu"] ul li,
div[role="option"] {
    color: #000000 !important;
    opacity: 1 !important;
    background-color: #ffffff !important;
}

/* Fix ALL text inside dropdown (force override) */
div[data-baseweb="popover"] * {
    color: #000000 !important;
    opacity: 1 !important;
}

            /* Selected value inside dropdown box -> BLACK */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    color: #000000 !important;
}

/* Ensure text inside is not faded */
section[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #000000 !important;
    opacity: 1 !important;
}

    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #000000 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #000000 !important;
}
                                          
</style>
""", unsafe_allow_html=True)

LABEL_MAP   = {0:"N (Normal)", 1:"S (Supraventricular)", 2:"V (Ventricular)", 3:"F (Fusion)", 4:"U (Unknown)"}
LABEL_SHORT = {0:"N", 1:"S", 2:"V", 3:"F", 4:"U"}
LABEL_DESC  = {
    "N": "Normal sinus rhythm — healthy heartbeat pattern",
    "S": "Supraventricular ectopy — irregular beats above ventricles",
    "V": "Ventricular ectopy — abnormal beats originating in ventricles",
    "F": "Fusion beat — combination of normal and ventricular beat",
    "U": "Unknown / Unclassifiable — ambiguous morphology",
}
LABEL_COLORS = {"N":"#00D4B8","S":"#4B8BFF","V":"#FF4B8B","F":"#FFB74D","U":"#B47BFF"}
CLASS_BADGE  = {"N":"badge-N","S":"badge-S","V":"badge-V","F":"badge-F","U":"badge-U"}
MODEL_COLORS = ["#00D4B8","#4B8BFF","#FF4B8B","#FFB74D","#B47BFF"]

def mpl_dark_style():
    plt.rcParams.update({
        "figure.facecolor" : "#111827",
        "axes.facecolor"   : "#1A2235",
        "axes.edgecolor"   : "#2A3A55",
        "axes.labelcolor"  : "#E6E315",
        "xtick.color"      : "#7A8BA3",
        "ytick.color"      : "#7A8BA3",
        "text.color"       : "#121315",
        "grid.color"       : "#2A3A55",
        "font.family"      : "monospace",
    })

mpl_dark_style()

DATA_JSON   = os.path.join(os.path.dirname(__file__), "ecg_data.json")
MODELS_PKL  = os.path.join(os.path.dirname(__file__), "ecg_models.pkl")
SCALER_PKL  = os.path.join(os.path.dirname(__file__), "ecg_scaler.pkl")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open(DATA_JSON,  "r") as f: data = json.load(f)
    with open(MODELS_PKL, "rb") as f: models = pickle.load(f)
    with open(SCALER_PKL, "rb") as f: scaler_info = pickle.load(f)
    return data, models, scaler_info

@st.cache_resource(show_spinner=False)
def train_from_df(df: pd.DataFrame):
    """Train full pipeline from a freshly uploaded DataFrame."""
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)
    feature_names = list(X.columns)
    X = X.fillna(X.mean())
    X_clipped = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_clipped)
    rf_sel = RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1)
    rf_sel.fit(Xs, y)
    importance = rf_sel.feature_importances_
    top50_idx = np.argsort(importance)[-50:][::-1]
    top50_names = [feature_names[i] for i in top50_idx]
    top50_imp   = [float(importance[i]) for i in top50_idx]
    X_sel = Xs[:, top50_idx]
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, stratify=y, random_state=42)
    model_defs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "SVM":                  SVC(kernel='rbf', C=5, gamma='scale', probability=True, random_state=42),
        "KNN":                  KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1),
        "Decision Tree":        DecisionTreeClassifier(max_depth=15, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    }
    results, conf_mats, trained = {}, {}, {}
    for name, model in model_defs.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy":  float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            "recall":    float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            "f1":        float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            "report":    classification_report(y_test, y_pred, target_names=["N","S","V","F","U"], output_dict=True, zero_division=0),
        }
        conf_mats[name] = confusion_matrix(y_test, y_pred).tolist()
        trained[name]   = model
    knn_k_vals, knn_train_acc, knn_test_acc = list(range(1,22,2)), [], []
    for k in knn_k_vals:
        km = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
        km.fit(X_train, y_train)
        knn_train_acc.append(float(km.score(X_train, y_train)))
        knn_test_acc.append(float(km.score(X_test, y_test)))
    scaler_info = {"scaler": scaler, "top50_idx": top50_idx, "feature_names": feature_names}
    data_payload = {
        "results": results, "conf_mats": conf_mats,
        "feature_names": feature_names, "top50_names": top50_names, "top50_imp": top50_imp,
        "top50_idx": top50_idx.tolist(), "knn_k_vals": knn_k_vals,
        "knn_train_acc": knn_train_acc, "knn_test_acc": knn_test_acc,
        "label_map": {str(k):v for k,v in LABEL_MAP.items()},
        "label_short": {str(k):v for k,v in LABEL_SHORT.items()},
        "class_dist": y.value_counts().sort_index().to_dict(),
    }
    return data_payload, trained, scaler_info

with st.sidebar:
    st.markdown('<div class="sidebar-sub">Heart Disease Classifier</div>', unsafe_allow_html=True)

    st.markdown("### Data Source")
    data_source = st.radio("Data Source", ["Use Pre-trained Models", "Upload New Dataset"], label_visibility="collapsed")

    uploaded_file = None
    if data_source == "Upload New Dataset":
        uploaded_file = st.file_uploader("Upload ECG file (Excel/CSV)", type=["xlsx","xls","csv"])

    st.markdown("---")
    st.markdown("### Model Selection")

    model_options = ["Logistic Regression","SVM","KNN","Decision Tree","Random Forest"]
    selected_model = st.selectbox("Active model for prediction", model_options, index=4)

    st.markdown("---")
    st.markdown("### Class Reference")
    for short, long in [("N","Normal"),("S","Supraventricular"),("V","Ventricular"),("F","Fusion"),("U","Unknown")]:
        st.markdown(f'<span class="badge {CLASS_BADGE[short]}">{short}</span> {long}', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built with Streamlit · sklearn · seaborn")

if uploaded_file is not None:
    with st.spinner("Training models on uploaded dataset…"):
        if uploaded_file.name.endswith(".csv"):
            df_up = pd.read_csv(uploaded_file)
        else:
            df_up = pd.read_excel(uploaded_file)
        if "label" not in df_up.columns:
            df_up.columns = [f"ecg_{i+1:03d}" for i in range(df_up.shape[1]-1)] + ["label"]
        data, trained_models, scaler_info = train_from_df(df_up)
    st.success("Models trained on uploaded dataset")
else:
    try:
        data, trained_models, scaler_info = load_artifacts()
    except FileNotFoundError:
        st.error("Pre-trained artifacts not found. Please upload your ecg_enhanced.xlsx file using the sidebar.")
        st.stop()

st.markdown("""
<div style="padding: 10px 0 24px;">
  <h1 style="margin:0; font-size:2.2rem;">ECG SIGNAL CLASSIFICATION AND VISUALIZATION FOR HEART DISEASE DETECTION</h1>
</div>
""", unsafe_allow_html=True)

tab1, = st.tabs(["Predict"])

with tab1:
    st.markdown('<div class="section-header">Real-time ECG Classification</div>', unsafe_allow_html=True)
    input_col, result_col = st.columns([1.2, 1])
    with input_col:
        st.markdown("#### Upload ECG Input")
        feature_names = data["feature_names"]
        top50_idx     = data["top50_idx"]
        scaler        = scaler_info["scaler"]
        input_vector_raw = None
        row_file = st.file_uploader(
            "Upload CSV file (single row with 194 features, no label)",
            type=["csv"]
        )
        if row_file:
            row_df = pd.read_csv(row_file, header=None)
            row_vals = row_df.values.flatten()
            if len(row_vals) >= len(feature_names):
                input_vector_raw = row_vals[:len(feature_names)]
                st.success(f"File loaded successfully ({len(input_vector_raw)} features)")
            else:
                st.error(f"Expected {len(feature_names)} features, got {len(row_vals)}")
    predict_btn = st.button("Run Prediction", use_container_width=True)
    with result_col:
        st.markdown("#### Prediction Result")
        if predict_btn and input_vector_raw is not None:
            raw_2d   = input_vector_raw.reshape(1, -1)
            if raw_2d.shape[1] == len(feature_names):
                scaled = scaler.transform(raw_2d)
            else:
                scaled = raw_2d  
            selected = scaled[:, top50_idx]

            model    = trained_models[selected_model]
            pred_cls = int(model.predict(selected)[0])
            pred_lbl = LABEL_SHORT[pred_cls]
            pred_desc = LABEL_DESC[pred_lbl]
            color    = LABEL_COLORS[pred_lbl]
            conf_str = ""
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(selected)[0]
                conf  = proba[pred_cls] * 100
                conf_str = f"Confidence: {conf:.1f}%"
            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label" style="color:{color};">{pred_lbl}</div>
                <div class="pred-desc">{LABEL_MAP[pred_cls]}</div>
                <div class="pred-conf">{conf_str}</div>
            </div>
            """, unsafe_allow_html=True)
            if hasattr(model, "predict_proba"):
                fig_prob, ax_prob = plt.subplots(figsize=(5, 2.5))
                classes = [LABEL_SHORT[i] for i in range(5)]
                colors_p = [LABEL_COLORS[c] for c in classes]
                bars = ax_prob.barh(classes, proba * 100, color=colors_p, height=0.5)
                ax_prob.set_xlabel("Probability (%)", fontsize=9)
                ax_prob.set_title("Class Probabilities", fontsize=10, color="#00D4B8")
                ax_prob.axvline(50, color="#FF4B8B", linestyle="--", alpha=0.4, linewidth=1)
                for bar, p in zip(bars, proba):
                    ax_prob.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                                 f"{p*100:.1f}%", va="center", fontsize=8)
                ax_prob.set_xlim(0, 110)
                fig_prob.tight_layout()
                st.pyplot(fig_prob, width="stretch")
                plt.close(fig_prob)
            st.markdown(f"""
            <div style="background:#1A2235;border-left:3px solid {color};border-radius:8px;padding:12px 16px;margin-top:12px;">
                <div style="font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;color:#7A8BA3;margin-bottom:6px;">Clinical Note</div>
                <div style="font-size:0.95rem;">{pred_desc}</div>
                <div style="font-size:0.8rem;color:#7A8BA3;margin-top:8px;">Model: <b style="color:#E8EDF5;">{selected_model}</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#1A2235;border:2px dashed rgba(0,212,184,0.2);border-radius:16px;
                        padding:40px;text-align:center;color:#7A8BA3;">
                <div style="font-size:3rem;">🫀</div>
                <div style="font-family:'Space Mono',monospace;margin-top:12px;">
                    Configure input & click<br><b style="color:#00D4B8;">Run Prediction</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    