import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd

# ── Page config (must be first) ──────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- global ---- */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2a2f3e;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* ---- metric cards ---- */
.metric-card {
    background: #1c2130;
    border: 1px solid #2a2f3e;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 12px;
}
.metric-card .label {
    font-size: 12px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 28px;
    font-weight: 700;
    color: #58a6ff;
}

/* ---- section cards ---- */
.section-card {
    background: #1c2130;
    border: 1px solid #2a2f3e;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}
.section-title {
    font-size: 13px;
    font-weight: 600;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2f3e;
}

/* ---- result banner ---- */
.result-churn {
    background: linear-gradient(135deg, #3d1a1a, #5c1f1f);
    border: 1px solid #f85149;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, #0d2818, #1a4731);
    border: 1px solid #3fb950;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}
.result-title { font-size: 26px; font-weight: 800; margin-bottom: 8px; }
.result-sub   { font-size: 14px; color: #8b949e; margin-top: 8px; }

/* ---- gauge bar ---- */
.gauge-wrap { margin: 16px 0; }
.gauge-bg {
    background: #2a2f3e;
    border-radius: 999px;
    height: 14px;
    overflow: hidden;
}
.gauge-fill {
    height: 14px;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ---- predict button ---- */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px 40px;
    font-size: 16px;
    font-weight: 700;
    width: 100%;
    cursor: pointer;
    letter-spacing: 0.5px;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85; }

/* ---- inputs ---- */
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] > div,
[data-testid="stSlider"] > div { color: #e0e0e0; }

label { color: #8b949e !important; font-size: 13px !important; }

/* ---- hide default streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'churn_model.pkl')

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['features']

model, features = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Churn Intelligence")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Navigation", ["🔮 Predict", "📊 Model Info"])
    st.markdown("---")
    n_estimators = model.n_estimators if hasattr(model, 'n_estimators') else 'N/A'
    n_features = len(features)
    st.markdown(f"""
    <div style='font-size:12px; color:#8b949e; line-height:1.8'>
    <b style='color:#58a6ff'>Model</b><br>Random Forest · {n_estimators} trees<br><br>
    <b style='color:#58a6ff'>Dataset</b><br>IBM Telco · 7,032 customers<br><br>
    <b style='color:#58a6ff'>AUC-ROC</b><br>0.824<br><br>
    <b style='color:#58a6ff'>Features</b><br>{n_features} customer attributes
    </div>
    """, unsafe_allow_html=True)

# ── Encoding helpers ──────────────────────────────────────────────────────────
binary_cols = ['Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Paperless Billing']

cat_options = {
    'Gender':            ['Male', 'Female'],
    'Multiple Lines':    ['No', 'Yes', 'No phone service'],
    'Internet Service':  ['DSL', 'Fiber optic', 'No'],
    'Online Security':   ['No', 'Yes', 'No internet service'],
    'Online Backup':     ['No', 'Yes', 'No internet service'],
    'Device Protection': ['No', 'Yes', 'No internet service'],
    'Tech Support':      ['No', 'Yes', 'No internet service'],
    'Streaming TV':      ['No', 'Yes', 'No internet service'],
    'Streaming Movies':  ['No', 'Yes', 'No internet service'],
    'Contract':          ['Month-to-month', 'One year', 'Two year'],
    'Payment Method':    ['Electronic check', 'Mailed check',
                          'Bank transfer (automatic)', 'Credit card (automatic)'],
}

def label_encode(val, options):
    return sorted(options).index(val)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if "🔮 Predict" in page:

    # Header
    st.markdown("""
    <div style='margin-bottom:28px'>
        <h1 style='color:#e6edf3; font-size:32px; font-weight:800; margin:0'>
            🔮 Customer Churn Predictor
        </h1>
        <p style='color:#8b949e; margin-top:6px; font-size:15px'>
            Enter customer details below to predict churn risk using our trained ML model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI strip
    k1, k2, k3, k4 = st.columns(4)
    for col, label, val in [
        (k1, "Model Accuracy", "79.5%"),
        (k2, "Customers Trained On", "7,032"),
        (k3, "Features Used", "19"),
        (k4, "Algorithm", "Random Forest"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='label'>{label}</div>
            <div class='value'>{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    inputs = {}

    # ── Section 1: Demographics ───────────────────────────────────────────────
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>👤 Customer Demographics</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        choice = st.selectbox("Gender", cat_options['Gender'])
        inputs['Gender'] = label_encode(choice, cat_options['Gender'])
    with c2:
        inputs['Senior Citizen'] = 1 if st.selectbox("Senior Citizen", ['No', 'Yes']) == 'Yes' else 0
    with c3:
        inputs['Partner'] = 1 if st.selectbox("Partner", ['No', 'Yes']) == 'Yes' else 0
    with c4:
        inputs['Dependents'] = 1 if st.selectbox("Dependents", ['No', 'Yes']) == 'Yes' else 0
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 2: Account & Billing ─────────────────────────────────────────
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>💳 Account & Billing</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        choice = st.selectbox("Contract Type", cat_options['Contract'])
        inputs['Contract'] = label_encode(choice, cat_options['Contract'])
    with c2:
        choice = st.selectbox("Payment Method", cat_options['Payment Method'])
        inputs['Payment Method'] = label_encode(choice, cat_options['Payment Method'])
    with c3:
        inputs['Paperless Billing'] = 1 if st.selectbox("Paperless Billing", ['No', 'Yes']) == 'Yes' else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        inputs['Tenure Months'] = st.slider("Tenure (months)", 0, 72, 12)
    with c2:
        inputs['Monthly Charges'] = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
    with c3:
        inputs['Total Charges'] = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0, step=10.0)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 3: Services ───────────────────────────────────────────────────
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📶 Services Subscribed</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        inputs['Phone Service'] = 1 if st.selectbox("Phone Service", ['No', 'Yes']) == 'Yes' else 0
        choice = st.selectbox("Multiple Lines", cat_options['Multiple Lines'])
        inputs['Multiple Lines'] = label_encode(choice, cat_options['Multiple Lines'])
        choice = st.selectbox("Internet Service", cat_options['Internet Service'])
        inputs['Internet Service'] = label_encode(choice, cat_options['Internet Service'])
    with c2:
        for feat in ['Online Security', 'Online Backup', 'Device Protection']:
            choice = st.selectbox(feat, cat_options[feat])
            inputs[feat] = label_encode(choice, cat_options[feat])
    with c3:
        for feat in ['Tech Support', 'Streaming TV', 'Streaming Movies']:
            choice = st.selectbox(feat, cat_options[feat])
            inputs[feat] = label_encode(choice, cat_options[feat])
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Predict button ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡  Run Churn Prediction"):
        row = np.array([[inputs[f] for f in features]])
        result = model.predict(row)[0]
        prob   = model.predict_proba(row)[0][1]
        pct    = int(prob * 100)

        bar_color = "#f85149" if result == 1 else "#3fb950"
        banner_class = "result-churn" if result == 1 else "result-safe"
        icon  = "🚨" if result == 1 else "✅"
        title = "High Churn Risk Detected" if result == 1 else "Customer Likely to Stay"
        sub   = "This customer shows strong indicators of churning." if result == 1 \
                else "This customer shows low churn risk based on their profile."

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='{banner_class}'>
            <div class='result-title'>{icon} {title}</div>
            <div style='font-size:48px; font-weight:900; color:{bar_color}; margin:12px 0'>{pct}%</div>
            <div class='gauge-wrap'>
                <div class='gauge-bg'>
                    <div class='gauge-fill' style='width:{pct}%; background:{bar_color}'></div>
                </div>
            </div>
            <div class='result-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

        # Breakdown columns
        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        b1.markdown(f"""<div class='metric-card'>
            <div class='label'>Churn Probability</div>
            <div class='value' style='color:{bar_color}'>{prob:.1%}</div>
        </div>""", unsafe_allow_html=True)
        b2.markdown(f"""<div class='metric-card'>
            <div class='label'>Retention Probability</div>
            <div class='value' style='color:#3fb950'>{1-prob:.1%}</div>
        </div>""", unsafe_allow_html=True)
        risk = "High" if pct >= 60 else ("Medium" if pct >= 35 else "Low")
        risk_color = "#f85149" if risk == "High" else ("#e3b341" if risk == "Medium" else "#3fb950")
        b3.markdown(f"""<div class='metric-card'>
            <div class='label'>Risk Level</div>
            <div class='value' style='color:{risk_color}'>{risk}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif "📊 Model Info" in page:

    st.markdown("""
    <div style='margin-bottom:28px'>
        <h1 style='color:#e6edf3; font-size:32px; font-weight:800; margin:0'>
            📊 Model Information
        </h1>
        <p style='color:#8b949e; margin-top:6px; font-size:15px'>
            Details about the trained model, dataset, and performance metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🤖 Model Details</div>", unsafe_allow_html=True)
        rows = [
            ("Algorithm",       "Random Forest Classifier"),
            ("Trees",           "100 estimators"),
            ("Train/Test Split","80% / 20%"),
            ("Random State",    "42 (reproducible)"),
            ("Target Column",   "Churn Value (0 = Stay, 1 = Churn)"),
        ]
        for k, v in rows:
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; padding:8px 0;
                        border-bottom:1px solid #2a2f3e; font-size:14px'>
                <span style='color:#8b949e'>{k}</span>
                <span style='color:#e6edf3; font-weight:600'>{v}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📈 Performance Metrics</div>", unsafe_allow_html=True)
        metrics = [
            ("Overall Accuracy", "79.5%",  "#58a6ff"),
            ("Precision (No Churn)", "83%","#3fb950"),
            ("Recall (No Churn)",    "90%","#3fb950"),
            ("Precision (Churn)",    "67%","#f85149"),
            ("Recall (Churn)",       "52%","#f85149"),
        ]
        for label, val, color in metrics:
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; padding:8px 0;
                        border-bottom:1px solid #2a2f3e; font-size:14px'>
                <span style='color:#8b949e'>{label}</span>
                <span style='color:{color}; font-weight:700'>{val}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Feature list
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧬 Features Used for Prediction</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, feat in enumerate(features):
        cols[i % 4].markdown(f"""
        <div style='background:#0f1117; border:1px solid #2a2f3e; border-radius:8px;
                    padding:8px 12px; margin-bottom:8px; font-size:13px; color:#c9d1d9'>
            {i+1}. {feat}
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Dataset summary
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🗂️ Dataset Summary</div>", unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    for col, label, val in [
        (d1, "Total Records",    "7,043"),
        (d2, "After Cleaning",   "7,032"),
        (d3, "Churn Rate",       "~26.5%"),
        (d4, "Source",           "IBM Telco"),
    ]:
        col.markdown(f"""<div class='metric-card'>
            <div class='label'>{label}</div>
            <div class='value' style='font-size:22px'>{val}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
