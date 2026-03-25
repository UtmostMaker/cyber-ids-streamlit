#!/usr/bin/env python3
"""
Dashboard IDS - Streamlit application avec 3 pages.
"""
import os, sys, json, time, random
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from explain import explain_prediction, get_feature_importance

# Config
st.set_page_config(
    page_title="Cyber IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palette cyber
COLOR_ATTACK = "#FF4B4B"
COLOR_NORMAL = "#00CC88"
COLOR_BG = "#0E1117"
COLOR_CARD = "#1E2530"
COLOR_TEXT = "#FAFAFA"
COLOR_ACCENT = "#4DA3FF"

sns.set_style("darkgrid")
plt.rcParams["figure.facecolor"] = COLOR_BG
plt.rcParams["axes.facecolor"] = COLOR_CARD
plt.rcParams["text.color"] = COLOR_TEXT
plt.rcParams["axes.labelcolor"] = COLOR_TEXT
plt.rcParams["xtick.color"] = COLOR_TEXT
plt.rcParams["ytick.color"] = COLOR_TEXT

# Charger donnees
@st.cache_data
def load_results():
    with open(os.path.join(BASE, "artifacts/results.json")) as f:
        return json.load(f)

@st.cache_data
def load_schema():
    with open(os.path.join(BASE, "artifacts/schema.json")) as f:
        return json.load(f)

@st.cache_data
def load_model():
    with open(os.path.join(BASE, "models/model.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_encoders():
    with open(os.path.join(BASE, "artifacts/label_encoders.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_test_data():
    with open(os.path.join(BASE, "artifacts/test.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_fi():
    return get_feature_importance()

def local_css():
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {COLOR_BG}; }}
    .stMetricValue, .stMetricLabel {{ color: {COLOR_TEXT} !important; }}
    .element-container {{ color: {COLOR_TEXT}; }}
    h1, h2, h3, h4 {{ color: {COLOR_TEXT} !important; }}
    .attack-badge {{
        background-color: {COLOR_ATTACK}; color: white;
        padding: 4px 12px; border-radius: 12px;
        font-weight: bold; display: inline-block;
    }}
    .normal-badge {{
        background-color: {COLOR_NORMAL}; color: white;
        padding: 4px 12px; border-radius: 12px;
        font-weight: bold; display: inline-block;
    }}
    .cyber-card {{
        background-color: {COLOR_CARD};
        border: 1px solid #333;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }}
    .stButton > button {{
        background-color: {COLOR_ACCENT};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 24px;
        font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True)

local_css()

# ========================
# PAGE 1 — Modeles
# ========================
def page_models():
    st.title("🛡️ Modeles IDS — Comparaison")
    results = load_results()
    schema = load_schema()
    fi = load_fi()
    X_test, y_test = load_test_data()
    model = load_model()

    col1, col2, col3, col4 = st.columns(4)
    rf = results["random_forest"]
    xgb = results["xgboost"]

    with col1:
        st.metric("RF Accuracy", f"{rf['accuracy']:.2%}")
    with col2:
        st.metric("RF F1", f"{rf['f1']:.2%}")
    with col3:
        st.metric("XGB Accuracy", f"{xgb['accuracy']:.2%}")
    with col4:
        st.metric("XGB F1", f"{xgb['f1']:.2%}")

    st.divider()

    # Bar chart comparatif
    col_metrics, col_importance = st.columns([1, 1])

    with col_metrics:
        st.subheader("Comparaison F1 / AUC-ROC")
        metrics_df = pd.DataFrame({
            "Metrique": ["Accuracy", "F1 Score", "AUC-ROC"],
            "Random Forest": [rf["accuracy"], rf["f1"], rf["auc_roc"]],
            "XGBoost": [xgb["accuracy"], xgb["f1"], xgb["auc_roc"]]
        })
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(3)
        w = 0.35
        bars1 = ax.bar(x - w/2, metrics_df["Random Forest"], w, label="Random Forest", color="#4DA3FF")
        bars2 = ax.bar(x + w/2, metrics_df["XGBoost"], w, label="XGBoost", color="#00CC88")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df["Metrique"])
        ax.set_ylim(0, 1.15)
        ax.legend()
        ax.set_facecolor(COLOR_CARD)
        fig.patch.set_facecolor(COLOR_BG)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Meilleur modele")
        st.success(f"**{results['best_model'].upper()}** (F1 = {results['best_f1']:.2%})")

    with col_importance:
        st.subheader("Importance des variables")
        fi_df = pd.DataFrame(fi).head(10)
        fi_df = fi_df.sort_values("importance")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(fi_df["feature"], fi_df["importance"], color=COLOR_ACCENT)
        ax.set_facecolor(COLOR_CARD)
        fig.patch.set_facecolor(COLOR_BG)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # Confusion matrices
    st.subheader("Matrices de confusion")
    col_rf, col_xgb = st.columns(2)

    with col_rf:
        st.markdown("**Random Forest**")
        cm_rf = np.array(rf["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Attaque"], yticklabels=["Normal", "Attaque"],
                    cbar=False)
        ax.set_xlabel("Predit")
        ax.set_ylabel("Reel")
        fig.patch.set_facecolor(COLOR_BG)
        st.pyplot(fig)
        plt.close(fig)

    with col_xgb:
        st.markdown("**XGBoost**")
        cm_xgb = np.array(xgb["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens", ax=ax,
                    xticklabels=["Normal", "Attaque"], yticklabels=["Normal", "Attaque"],
                    cbar=False)
        ax.set_xlabel("Predit")
        ax.set_ylabel("Reel")
        fig.patch.set_facecolor(COLOR_BG)
        st.pyplot(fig)
        plt.close(fig)

# ========================
# PAGE 2 — Simulation RT
# ========================
def page_simulation():
    st.title("📡 Simulation Temps Reel")
    st.markdown("Lancement d'une simulation de sessions reseau avec detection d'intrusion.")

    schema = load_schema()
    encoders = load_encoders()
    model = load_model()
    fi_data = load_fi()

    # Preparer des sessions synthetiques
    def generate_session():
        is_attack = random.random() < 0.30  # 30% attaques
        if is_attack:
            session = {
                "duration": random.randint(0, 10),
                "src_bytes": random.randint(0, 500),
                "dst_bytes": random.randint(0, 500),
                "protocol_type": random.choice(["TCP", "UDP", "ICMP"]),
                "service": random.choice(["other", "telnet", "ftp", "smtp"]),
                "flag": random.choice(["S0", "REJ", "RSTR", "SH"]),
                "land": random.choice([0, 1]),
                "wrong_fragment": random.randint(0, 3),
                "urgent": random.randint(0, 2),
                "hot": random.randint(3, 10),
                "logged_in": 0,
                "num_compromised": random.randint(5, 15),
                "count": random.randint(80, 200),
                "srv_count": random.randint(0, 20),
                "serror_rate": random.uniform(0.6, 1.0),
                "srv_serror_rate": random.uniform(0.5, 1.0),
                "rerror_rate": random.uniform(0.3, 0.8),
                "srv_rerror_rate": random.uniform(0.3, 0.8),
                "diff_srv_rate": random.uniform(0.0, 0.2),
                "dst_host_count": random.randint(0, 80),
            }
        else:
            session = {
                "duration": random.randint(100, 5000),
                "src_bytes": random.randint(1000, 2000000),
                "dst_bytes": random.randint(2000, 3000000),
                "protocol_type": random.choice(["TCP", "UDP"]),
                "service": random.choice(["http", "https", "ssh", "dns"]),
                "flag": random.choice(["SF", "S0"]),
                "land": 0,
                "wrong_fragment": 0,
                "urgent": 0,
                "hot": random.randint(0, 2),
                "logged_in": 1,
                "num_compromised": random.randint(0, 3),
                "count": random.randint(5, 50),
                "srv_count": random.randint(5, 40),
                "serror_rate": random.uniform(0.0, 0.1),
                "srv_serror_rate": random.uniform(0.0, 0.1),
                "rerror_rate": random.uniform(0.0, 0.05),
                "srv_rerror_rate": random.uniform(0.0, 0.05),
                "diff_srv_rate": random.uniform(0.1, 0.5),
                "dst_host_count": random.randint(100, 250),
            }
        return session, "attaque" if is_attack else "normal"

    if "simulation_running" not in st.session_state:
        st.session_state.simulation_running = False
    if "sim_counter" not in st.session_state:
        st.session_state.sim_counter = {"attacks": 0, "total": 0, "sessions": []}

    col_btn, col_reset = st.columns([1, 1])
    with col_btn:
        if st.button("🚀 Lancer la simulation", use_container_width=True):
            st.session_state.simulation_running = True
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.sim_counter = {"attacks": 0, "total": 0, "sessions": []}
            st.rerun()

    # Compteurs
    col_a, col_t, col_r = st.columns(3)
    with col_a:
        st.metric("Attaques detectees", st.session_state.sim_counter["attacks"], delta_color="inverse")
    with col_t:
        st.metric("Sessions traitees", st.session_state.sim_counter["total"])
    with col_r:
        if st.session_state.sim_counter["total"] > 0:
            rate = st.session_state.sim_counter["attacks"] / st.session_state.sim_counter["total"]
            st.metric("Taux d'attaques", f"{rate:.1%}")

    st.divider()

    # Zone de simulation
    placeholder = st.empty()

    if st.session_state.simulation_running:
        for i in range(20):  # 20 sessions
            session, true_label = generate_session()
            result = explain_prediction(session)
            pred = result["prediction"]
            confiance = result["confiance"]

            st.session_state.sim_counter["total"] += 1
            if pred == "attaque":
                st.session_state.sim_counter["attacks"] += 1

            badge_color = COLOR_ATTACK if pred == "attaque" else COLOR_NORMAL
            badge = f'<span class="attack-badge">ATTAQUE</span>' if pred == "attaque" else f'<span class="normal-badge">NORMAL</span>'

            with placeholder.container():
                st.markdown(f"""
                <div class="cyber-card">
                    <strong>Session #{i+1}</strong> &nbsp;|&nbsp;
                    Proto: {session['protocol_type']} &nbsp;|&nbsp;
                    Service: {session['service']} &nbsp;|&nbsp;
                    Flag: {session['flag']} &nbsp;|&nbsp;
                    src_bytes: {session['src_bytes']} &nbsp;|&nbsp;
                    serror_rate: {session['serror_rate']:.2f} &nbsp;|&nbsp;
                    {badge} &nbsp;|&nbsp;
                    Confiance: {confiance:.0%}
                </div>
                """, unsafe_allow_html=True)

            time.sleep(0.5)

        st.session_state.simulation_running = False
        st.success("Simulation terminee. Cliquez a nouveau sur 'Lancer la simulation' pour recommencer.")

# ========================
# PAGE 3 — Tester une session
# ========================
def page_test_session():
    st.title("🔍 Tester une session")
    st.markdown("Saisissez les caracteristiques d'une session reseau pour analyser le risque.")

    schema = load_schema()
    encoders = load_encoders()
    feature_names = schema["feature_names"]
    categorical = schema["categorical_cols"]

    # Valeurs par defaut
    defaults = {
        "duration": 1000, "src_bytes": 500000, "dst_bytes": 800000,
        "protocol_type": "TCP", "service": "http", "flag": "SF",
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 1,
        "logged_in": 1, "num_compromised": 2, "count": 30,
        "srv_count": 25, "serror_rate": 0.05, "srv_serror_rate": 0.05,
        "rerror_rate": 0.01, "srv_rerror_rate": 0.01,
        "diff_srv_rate": 0.2, "dst_host_count": 180
    }

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Parametres de connexion**")
        duration = st.number_input("Duration (sec)", 0, 100000, 1000)
        protocol_type = st.selectbox("Protocol", encoders["protocol_type"].classes_)
        service = st.selectbox("Service", encoders["service"].classes_)
        flag = st.selectbox("Flag", encoders["flag"].classes_)
        land = st.selectbox("Land", [0, 1])
        logged_in = st.selectbox("Logged In", [0, 1])

    with col2:
        st.markdown("**Volumes et taux**")
        src_bytes = st.number_input("Src Bytes", 0, 10000000, 500000)
        dst_bytes = st.number_input("Dst Bytes", 0, 10000000, 800000)
        serror_rate = st.slider("Serror Rate", 0.0, 1.0, 0.05)
        rerror_rate = st.slider("Rerror Rate", 0.0, 1.0, 0.01)
        srv_serror_rate = st.slider("Srv Serror Rate", 0.0, 1.0, 0.05)
        count = st.number_input("Count", 0, 500, 30)
        srv_count = st.number_input("Srv Count", 0, 500, 25)
        num_compromised = st.number_input("Num Compromised", 0, 50, 2)
        diff_srv_rate = st.slider("Diff Srv Rate", 0.0, 1.0, 0.2)
        dst_host_count = st.number_input("Dst Host Count", 0, 255, 180)
        hot = st.slider("Hot", 0, 10, 1)
        wrong_fragment = st.number_input("Wrong Fragment", 0, 10, 0)
        urgent = st.number_input("Urgent", 0, 10, 0)
        srv_rerror_rate = st.slider("Srv Rerror Rate", 0.0, 1.0, 0.01)

    session_data = {
        "duration": duration, "src_bytes": src_bytes, "dst_bytes": dst_bytes,
        "protocol_type": protocol_type, "service": service, "flag": flag,
        "land": land, "logged_in": logged_in, "serror_rate": serror_rate,
        "rerror_rate": rerror_rate, "srv_serror_rate": srv_serror_rate,
        "count": count, "srv_count": srv_count, "num_compromised": num_compromised,
        "diff_srv_rate": diff_srv_rate, "dst_host_count": dst_host_count,
        "hot": hot, "wrong_fragment": wrong_fragment, "urgent": urgent,
        "srv_rerror_rate": srv_rerror_rate
    }

    st.divider()

    if st.button("🔬 Analyser cette session", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            result = explain_prediction(session_data)

        pred = result["prediction"]
        confiance = result["confiance"]

        col_pred, col_conf = st.columns(2)
        with col_pred:
            if pred == "attaque":
                st.error(f"🚨 **ATTAQUE DETECTEE**")
            else:
                st.success(f"✅ **SESSION NORMALE**")

        with col_conf:
            st.metric("Confiance", f"{confiance:.1%}")

        # Bar de confiance
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.barh(["Normal", "Attaque"],
                [result["probabilites"]["normal"], result["probabilites"]["attaque"]],
                color=[COLOR_NORMAL, COLOR_ATTACK])
        ax.set_xlim(0, 1)
        ax.set_facecolor(COLOR_CARD)
        fig.patch.set_facecolor(COLOR_BG)
        st.pyplot(fig)
        plt.close(fig)

        # Top facteurs
        st.subheader("Facteurs cles de la decision")
        for i, f in enumerate(result["top_facteurs"], 1):
            sens_emoji = "⬆️" if f["sens"] == "augmente" else "⬇️"
            sens_color = COLOR_ATTACK if pred == "attaque" else COLOR_NORMAL
            st.markdown(f"{i}. **{f['feature']}** {sens_emoji} — {f['sens']} le risque")
            st.caption(f"   Valeur: {f.get('valeur', f.get('contribution', 0)):.4f} | Contribution: {f.get('contribution', f.get('importance', 0)):.4f}")

        st.info(result["explication"])

# ========================
# SIDEBAR NAVIGATION
# ========================
st.sidebar.title("🛡️ Cyber IDS")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["📊 Modeles", "📡 Simulation RT", "🔍 Tester une session"])

if page == "📊 Modeles":
    page_models()
elif page == "📡 Simulation RT":
    page_simulation()
else:
    page_test_session()
