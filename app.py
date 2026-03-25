#!/usr/bin/env python3
"""
Cyber IDS - Dashboard Streamlit
Application d'analyse d'intrusion en temps reel avec ML.
Support multi-dataset: NSL-KDD et CICIDS2017.
"""
import os, sys, json, pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE, "artifacts")
MODELS = os.path.join(BASE, "models")
DATA = os.path.join(BASE, "data")

# ── Chargement des artefacts (avec cache) ─────────────────────────────────────
@st.cache_data
def load_results(dataset):
    if dataset == "cicids2017":
        path = os.path.join(ARTIFACTS, "results_cicids2017.json")
    else:
        path = os.path.join(ARTIFACTS, "results.json")
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_feature_importance(dataset):
    if dataset == "cicids2017":
        return pd.read_csv(os.path.join(ARTIFACTS, "feature_importance_cicids2017.csv"))
    return pd.read_csv(os.path.join(ARTIFACTS, "feature_importance.csv"))

@st.cache_data
def load_schema(dataset):
    if dataset == "cicids2017":
        path = os.path.join(ARTIFACTS, "schema_cicids2017.json")
    else:
        path = os.path.join(ARTIFACTS, "schema.json")
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_model_and_preprocessor(dataset):
    if dataset == "cicids2017":
        model_path = os.path.join(MODELS, "model_cicids2017.pkl")
        preproc_path = os.path.join(ARTIFACTS, "preprocessor_cicids2017.pkl")
        le_path = None
    else:
        model_path = os.path.join(MODELS, "model.pkl")
        preproc_path = os.path.join(ARTIFACTS, "preprocessor.pkl")
        le_path = os.path.join(ARTIFACTS, "label_encoders.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(preproc_path, "rb") as f:
        preprocessor = pickle.load(f)
    if le_path:
        with open(le_path, "rb") as f:
            label_encoders = pickle.load(f)
    else:
        label_encoders = {}
    return model, preprocessor, label_encoders

# ── Config theme sombre ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cyber IDS",
    layout="wide",
    page_icon="🛡️",
)
dark_css = """
<style>
    .stApp { background: #0d1117; color: #c9d1d9; }
    h1, h2, h3 { color: #58a6ff; }
    .attack { color: #FF4B4B; font-weight: bold; }
    .normal { color: #00CC88; font-weight: bold; }
    .stMetric { background: #161b22; border-radius: 8px; padding: 12px; }
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input { background: #161b22; color: #c9d1d9; }
    div[data-testid="stHorizontalBlock"] > div { background: #161b22; border-radius: 8px; padding: 12px; }
    .stTabs > div > div { background: #161b22; }
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🛡️ Cyber IDS")
st.sidebar.caption("Systeme de Detection d'Intrusion")
page = st.sidebar.radio(
    "Navigation",
    ["1. Modeles", "2. Simulation RT", "3. Tester une session", "4. Live Stream"],
    index=0
)

# ── Dataset selector in sidebar ───────────────────────────────────────────────
available_datasets = []
if os.path.exists(os.path.join(ARTIFACTS, "results.json")):
    available_datasets.append(("NSL-KDD", "nslkdd"))
if os.path.exists(os.path.join(ARTIFACTS, "results_cicids2017.json")):
    available_datasets.append(("CICIDS2017", "cicids2017"))

dataset_options = [d[0] for d in available_datasets]
selected_dataset_label = st.sidebar.selectbox(
    "Dataset actif",
    dataset_options,
    index=0 if dataset_options else 0
)
active_dataset = dict(available_datasets).get(selected_dataset_label, "nslkdd")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1  -  Modeles
# ══════════════════════════════════════════════════════════════════════════════
if page == "1. Modeles":
    st.title("Comparaison des Modeles")
    st.caption(f"Dataset actif: {selected_dataset_label}")

    try:
        results = load_results(active_dataset)
        features = load_feature_importance(active_dataset)
        schema = load_schema(active_dataset)
    except Exception as e:
        st.error(f"Erreur de chargement des artefacts ({active_dataset}): {e}")
        st.stop()

    rf = results["random_forest"]
    xgb = results["xgboost"]

    # ── Métriques principales ────────────────────────────────────────────────
    st.subheader("Performances")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy RF", f"{rf['accuracy']*100:.2f}%")
    with col2:
        st.metric("F1 RF", f"{rf['f1']*100:.2f}%")
    with col3:
        st.metric("Accuracy XGB", f"{xgb['accuracy']*100:.2f}%")
    with col4:
        st.metric("F1 XGB", f"{xgb['f1']*100:.2f}%")

    # ── AUC-ROC ────────────────────────────────────────────────────────────────
    col_auc1, col_auc2 = st.columns(2)
    with col_auc1:
        st.metric("AUC-ROC RF", f"{rf['auc_roc']*100:.2f}%")
    with col_auc2:
        st.metric("AUC-ROC XGB", f"{xgb['auc_roc']*100:.2f}%")

    st.markdown("---")

    # ── Bar chart comparatif ─────────────────────────────────────────────────
    st.subheader("Comparaison RF vs XGBoost")
    metrics_df = pd.DataFrame({
        "Metrique": ["Accuracy", "F1-Score", "AUC-ROC"],
        "Random Forest": [rf["accuracy"], rf["f1"], rf["auc_roc"]],
        "XGBoost": [xgb["accuracy"], xgb["f1"], xgb["auc_roc"]]
    })
    fig_bar = go.Figure()
    for col in ["Random Forest", "XGBoost"]:
        fig_bar.add_trace(go.Bar(
            name=col, x=metrics_df["Metrique"],
            y=metrics_df[col], text=[f"{v*100:.1f}%" for v in metrics_df[col]],
            textposition="outside"
        ))
    fig_bar.update_layout(
        barmode="group", template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        legend=dict(bgcolor="#161b22", font=dict(color="#c9d1d9")),
        height=400
    )
    st.plotly_chart(fig_bar, width='stretch')

    st.markdown("---")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    st.subheader("Matrices de Confusion")
    col_rf, col_xgb = st.columns(2)
    with col_rf:
        st.markdown("**Random Forest**")
        cm_rf = np.array(rf["confusion_matrix"])
        fig_cm_rf = px.imshow(
            cm_rf, x=["Normal", "Attaque"], y=["Normal", "Attaque"],
            color_continuous_scale=["#0d1117", "#FF4B4B"], text_auto=True
        )
        fig_cm_rf.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            height=350, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_cm_rf, width='stretch')

    with col_xgb:
        st.markdown("**XGBoost**")
        cm_xgb = np.array(xgb["confusion_matrix"])
        fig_cm_xgb = px.imshow(
            cm_xgb, x=["Normal", "Attaque"], y=["Normal", "Attaque"],
            color_continuous_scale=["#0d1117", "#FF4B4B"], text_auto=True
        )
        fig_cm_xgb.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            height=350, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_cm_xgb, width='stretch')

    st.markdown("---")

    # ── Feature importance ───────────────────────────────────────────────────
    st.subheader("Importance des Variables (Meilleur Modele)")
    top_n = st.slider("Nombre de features", 5, 20, 10, key="fi_slider")
    top_feat = features.head(top_n)
    fig_fi = px.bar(
        top_feat, x="importance", y="feature", orientation="h",
        title=f"Top {top_n} variables", color="importance",
        color_continuous_scale=["#4B9FFF", "#FF4B4B"]
    )
    fig_fi.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        height=500, font=dict(color="#c9d1d9"),
        coloraxis_colorbar=dict(title="Importance", tickcolor="#c9d1d9")
    )
    st.plotly_chart(fig_fi, width='stretch')

    st.markdown("---")

    # ── Schema du dataset ─────────────────────────────────────────────────────
    with st.expander("Schema du dataset"):
        st.json(schema)

    # ── Comparaison des datasets (si les deux sont disponibles) ──────────────
    if len(available_datasets) > 1:
        with st.expander("Comparaison NSL-KDD vs CICIDS2017"):
            st.markdown("### Comparaison des deux datasets")
            comparison_data = []
            for label, ds in available_datasets:
                try:
                    r = load_results(ds)
                    comparison_data.append({
                        "Dataset": label,
                        "Modele": "XGBoost",
                        "Accuracy": f"{r['xgboost']['accuracy']*100:.2f}%",
                        "F1": f"{r['xgboost']['f1']*100:.2f}%",
                        "AUC-ROC": f"{r['xgboost']['auc_roc']*100:.2f}%"
                    })
                except:
                    pass
            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data), hide_index=True)

                fig_comp = go.Figure()
                for row in comparison_data:
                    fig_comp.add_trace(go.Bar(
                        name=row["Dataset"],
                        x=["Accuracy", "F1", "AUC-ROC"],
                        y=[float(row[k].replace("%",""))/100 for k in ["Accuracy","F1","AUC-ROC"]],
                        text=[row[k] for k in ["Accuracy","F1","AUC-ROC"]],
                        textposition="outside"
                    ))
                fig_comp.update_layout(
                    barmode="group", template="plotly_dark",
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    height=350,
                    legend=dict(bgcolor="#161b22", font=dict(color="#c9d1d9"))
                )
                st.plotly_chart(fig_comp, width='stretch')
            else:
                st.warning("Impossible de charger les metriques pour la comparaison.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2  -  Simulation RT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "2. Simulation RT":
    st.title("Simulation Temps Reel")
    st.caption(f"Dataset actif: {selected_dataset_label} | Generation de sessions reseau pour tester le systeme IDS")

    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False
    if "sim_counter" not in st.session_state:
        st.session_state.sim_counter = 0
    if "sim_attacks" not in st.session_state:
        st.session_state.sim_attacks = 0
    if "sim_total" not in st.session_state:
        st.session_state.sim_total = 0
    if "sim_history" not in st.session_state:
        st.session_state.sim_history = []

    col_a, col_b = st.columns(2)
    counter_placeholder = col_a.empty()
    rate_placeholder = col_b.empty()
    st.subheader("Sessions en cours")
    feed_placeholder = st.empty()

    # Chargement modele et preprocessor
    try:
        model, preprocessor, label_encoders = load_model_and_preprocessor(active_dataset)
    except Exception as e:
        st.error(f"Impossible de charger le modele {active_dataset}: {e}")
        st.stop()

    # ── Generateur de sessions selon le dataset ─────────────────────────────
    def gen_session_cicids2017(idx):
        """Genere une session basee sur les statistiques CICIDS2017."""
        import random
        normal = random.random() > 0.40  # 40% attaques
        stealth = random.random() > 0.5   # 50% stealth attacks
        if normal:
            duration = np.clip(np.random.lognormal(4.5, 2), 0, 10000)
            fwd_packets = random.poisson(8 if not stealth else 5)
            bwd_packets = random.poisson(6 if not stealth else 3)
            fwd_len = int(np.clip(np.random.lognormal(9, 2), 0, 1e8))
            bwd_len = int(np.clip(np.random.lognormal(10, 2), 0, 1e8))
            flow_bytes_s = np.clip(np.random.lognormal(12, 2), 0, 1e10)
            flow_packets_s = np.clip(np.random.lognormal(3, 1.5), 0, 1e5)
            fwd_iat_mean = np.clip(np.random.lognormal(5, 2), 0, 1e6)
            bwd_iat_mean = np.clip(np.random.lognormal(6, 2), 0, 1e6)
            syn_count = random.poisson(2 if not stealth else 5)
            rst_count = random.poisson(0.5)
            ack_count = int(np.clip(np.random.lognormal(3, 1), 0, 1e5))
            psh_count = random.poisson(1)
            active_mean = np.clip(np.random.lognormal(7, 2), 0, 1e6)
            idle_mean = np.clip(np.random.lognormal(8, 2), 0, 1e6)
            flow_iat_mean = np.clip(np.random.lognormal(5.5, 2), 0, 1e6)
            fwd_len_mean = np.clip(np.random.lognormal(8, 2), 0, 1e6)
            bwd_len_mean = np.clip(np.random.lognormal(9, 2), 0, 1e6)
            dst_port = random.choice([80, 443, 22, 53, 8080, 21])
            is_attack = False
        else:
            # Attaques avec 50% stealth
            duration = np.clip(np.random.lognormal(1 if not stealth else 4, 1.5 if not stealth else 2), 0, 10000)
            fwd_packets = random.poisson(3 if not stealth else 8)
            bwd_packets = random.poisson(2 if not stealth else 6)
            fwd_len = int(np.clip(np.random.lognormal(4 if not stealth else 9, 2), 0, 1e8))
            bwd_len = int(np.clip(np.random.lognormal(4 if not stealth else 10, 2), 0, 1e8))
            flow_bytes_s = np.clip(np.random.lognormal(8 if not stealth else 12, 2), 0, 1e10)
            flow_packets_s = np.clip(np.random.lognormal(5 if not stealth else 3, 2), 0, 1e5)
            fwd_iat_mean = np.clip(np.random.lognormal(2 if not stealth else 5, 1.5 if not stealth else 2), 0, 1e6)
            bwd_iat_mean = np.clip(np.random.lognormal(2 if not stealth else 6, 1.5 if not stealth else 2), 0, 1e6)
            syn_count = random.poisson(8 if not stealth else 2)
            rst_count = random.poisson(1 if not stealth else 0.5)
            ack_count = int(np.clip(np.random.lognormal(2 if not stealth else 4, 1), 0, 1e5))
            psh_count = random.poisson(2 if not stealth else 1)
            active_mean = np.clip(np.random.lognormal(1 if not stealth else 7, 1 if not stealth else 2), 0, 1e6)
            idle_mean = np.clip(np.random.lognormal(0.5 if not stealth else 8, 0.5 if not stealth else 2), 0, 1e6)
            flow_iat_mean = np.clip(np.random.lognormal(1.5 if not stealth else 5.5, 1 if not stealth else 2), 0, 1e6)
            fwd_len_mean = np.clip(np.random.lognormal(3 if not stealth else 8, 1.5 if not stealth else 2), 0, 1e6)
            bwd_len_mean = np.clip(np.random.lognormal(3 if not stealth else 9, 1.5 if not stealth else 2), 0, 1e6)
            dst_port = random.choice([22, 21, 80, 443, 0, 53, 8080])
            is_attack = True

        fwd_psh_flags = random.poisson(1)
        return {
            "session_id": f"SES-{idx:05d}",
            "src_ip": f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
            "dst_ip": f"10.0.{random.randint(1,254)}.{random.randint(1,254)}",
            "src_port": random.randint(1024, 65535),
            "dst_port": dst_port,
            "protocol": "TCP",
            "duration": round(duration, 2),
            "fwd_packets": fwd_packets,
            "bwd_packets": bwd_packets,
            "fwd_len": fwd_len,
            "bwd_len": bwd_len,
            "flow_bytes_s": round(flow_bytes_s, 2),
            "flow_packets_s": round(flow_packets_s, 4),
            "fwd_iat_mean": round(fwd_iat_mean, 2),
            "bwd_iat_mean": round(bwd_iat_mean, 2),
            "fwd_psh_flags": fwd_psh_flags,
            "syn_count": syn_count,
            "rst_count": rst_count,
            "ack_count": ack_count,
            "psh_count": psh_count,
            "active_mean": round(active_mean, 2),
            "idle_mean": round(idle_mean, 2),
            "flow_iat_mean": round(flow_iat_mean, 2),
            "fwd_len_mean": round(fwd_len_mean, 2),
            "bwd_len_mean": round(bwd_len_mean, 2),
            "_is_attack": is_attack
        }

    def gen_session_nslkdd(idx):
        """Genere une session basee sur les statistiques NSL-KDD."""
        import random
        normal = random.random() > 0.42
        protocols = ["tcp", "udp", "icmp"]
        flags = ["SF", "S0", "REJ", "RSTR", "SH"]
        services = ["http", "ftp", "smtp", "dns", "ssh", "irc", "telnet", "pop3"]
        src_ip = f"192.168.{random.randint(1,254)}.{random.randint(1,254)}"
        dst_ip = f"10.0.{random.randint(1,254)}.{random.randint(1,254)}"
        if normal:
            duration = np.clip(np.random.lognormal(2.5, 1.5), 0, 10000)
            src_bytes = int(np.clip(np.random.lognormal(8, 3), 0, 1e8))
            dst_bytes = int(np.clip(np.random.lognormal(10, 2), 0, 1e9))
            serror_rate = np.clip(np.random.exponential(0.1), 0, 1)
            logged_in = random.choice([0, 1])
            protocol = random.choices(protocols, [0.8, 0.15, 0.05])[0]
        else:
            duration = np.clip(np.random.exponential(1.5), 0, 100)
            src_bytes = int(np.clip(np.random.exponential(5000), 0, 1e7))
            dst_bytes = int(np.clip(np.random.exponential(200), 0, 1e5))
            serror_rate = np.clip(np.random.uniform(0.3, 1.0), 0, 1)
            logged_in = random.randint(0, 1)
            protocol = random.choices(protocols, [0.6, 0.3, 0.1])[0]
        return {
            "session_id": f"SES-{idx:05d}",
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": random.randint(1024, 65535),
            "dst_port": random.choice([80, 443, 22, 21, 25, 53, 8080, 3389]),
            "protocol": protocol,
            "duration": round(duration, 2),
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "serror_rate": round(serror_rate, 3),
            "logged_in": logged_in,
            "count": random.randint(0, 100),
            "srv_count": random.randint(0, 100),
            "rerror_rate": round(np.clip(np.random.exponential(0.1), 0, 1), 3),
            "service": random.choice(services),
            "flag": random.choice(flags),
            "wrong_fragment": random.randint(0, 3),
            "urgent": random.randint(0, 2),
            "hot": random.randint(0, 10),
            "num_compromised": random.randint(0, 8) if not normal else random.randint(0, 1),
            "diff_srv_rate": round(np.clip(np.random.exponential(0.2), 0, 1), 3),
            "dst_host_count": random.randint(0, 255),
            "_is_attack": not normal
        }

    def gen_session(idx):
        if active_dataset == "cicids2017":
            return gen_session_cicids2017(idx)
        return gen_session_nslkdd(idx)

    def predict_session(session):
        try:
            import explain
            pred, conf, factors = explain.explain_prediction(session, dataset=active_dataset)
            return pred, conf, factors
        except Exception as e:
            return "unknown", 0.5, []

    col_start, col_stop = st.columns(2)
    if col_start.button("▶ Lancer la simulation", type="primary", width='stretch'):
        st.session_state.sim_running = True
        st.session_state.sim_counter = 0
        st.session_state.sim_attacks = 0
        st.session_state.sim_total = 0
        st.session_state.sim_history = []

    if col_stop.button("■ Arreter", width='stretch'):
        st.session_state.sim_running = False

    if st.session_state.sim_running:
        counter_placeholder.metric("Sessions analysees", st.session_state.sim_counter)
        rate_placeholder.metric("Attaques detectees", st.session_state.sim_attacks,
                                delta=f"{st.session_state.sim_attacks/st.max(st.session_state.sim_counter,1)*100:.1f}%")

        import time
        session = gen_session(st.session_state.sim_counter + 1)
        pred, conf, factors = predict_session(session)
        st.session_state.sim_counter += 1
        if pred == "ATTACK" or session["_is_attack"]:
            st.session_state.sim_attacks += 1
        st.session_state.sim_total += 1

        is_attack = pred == "ATTACK" or session["_is_attack"]
        badge = "🔴 ATTAQUE" if is_attack else "🟢 Normal"

        st.session_state.sim_history.insert(0, {
            "id": session["session_id"],
            "src": session["src_ip"],
            "proto": session["protocol"],
            "dur": f"{session.get('duration', session.get('fwd_packets', 0))}s",
            "badge": badge,
            "conf": f"{conf*100:.1f}%" if pred != "unknown" else "N/A",
            "pred": pred
        })

        with feed_placeholder.container():
            rows = st.session_state.sim_history[:20]
            df = pd.DataFrame(rows)
            st.dataframe(
                df[["id", "src", "proto", "dur", "badge", "conf"]],
                width='stretch', hide_index=True,
                column_config={"badge": st.column_config.TextColumn("Statut"),
                               "conf": st.column_config.TextColumn("Confiance")}
            )

        time.sleep(0.8)
        st.rerun()
    else:
        if st.session_state.sim_counter > 0:
            st.success(f"Simulation terminee : {st.session_state.sim_counter} sessions, {st.session_state.sim_attacks} attaques detectees.")
        else:
            st.info("Appuyez sur 'Lancer la simulation' pour commencer.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3  -  Tester une session
# ══════════════════════════════════════════════════════════════════════════════
elif page == "3. Tester une session":
    st.title("Tester une Session Manuellement")
    st.caption(f"Dataset actif: {selected_dataset_label}")

    try:
        model, preprocessor, label_encoders = load_model_and_preprocessor(active_dataset)
    except Exception as e:
        st.error(f"Impossible de charger le modele {active_dataset}: {e}")
        st.stop()

    col1, col2 = st.columns([1, 1])

    if active_dataset == "cicids2017":
        with col1:
            st.subheader("Parametres de la session (CICIDS2017)")
            with st.form("session_form"):
                src_ip = st.text_input("IP Source", "192.168.1.105")
                dst_ip = st.text_input("IP Destination", "10.0.0.1")
                src_port = st.number_input("Port Source", 1024, 65535, 54321)
                dst_port = st.number_input("Port Destination", 1, 65535, 80)
                protocol = st.selectbox("Protocol", ["tcp", "udp", "icmp"])
                duration = st.number_input("Flow Duration (s)", 0.0, 10000.0, 5.0)
                fwd_packets = st.number_input("Total Fwd Packets", 1, 10000, 8)
                bwd_packets = st.number_input("Total Bwd Packets", 0, 10000, 6)
                fwd_len = st.number_input("Total Fwd Packet Length", 0, 10000000, 5000)
                bwd_len = st.number_input("Total Bwd Packet Length", 0, 10000000, 3000)
                flow_bytes_s = st.number_input("Flow Bytes/s", 0.0, 1e10, 1e6)
                flow_packets_s = st.number_input("Flow Packets/s", 0.0, 1e6, 100.0)
                fwd_iat_mean = st.number_input("Fwd IAT Mean (s)", 0.0, 1e6, 0.5)
                bwd_iat_mean = st.number_input("Bwd IAT Mean (s)", 0.0, 1e6, 0.5)
                fwd_psh_flags = st.number_input("Fwd PSH Flags", 0, 100, 1)
                syn_count = st.number_input("SYN Flag Count", 0, 1000, 2)
                rst_count = st.number_input("RST Flag Count", 0, 100, 0)
                ack_count = st.number_input("ACK Flag Count", 0, 1000, 5)
                psh_count = st.number_input("PSH Flag Count", 0, 100, 1)
                active_mean = st.number_input("Active Mean (s)", 0.0, 1e6, 0.3)
                idle_mean = st.number_input("Idle Mean (s)", 0.0, 1e6, 0.5)
                flow_iat_mean = st.number_input("Flow IAT Mean (s)", 0.0, 1e6, 0.5)
                fwd_len_mean = st.number_input("Fwd Packet Length Mean", 0.0, 1e6, 500.0)
                bwd_len_mean = st.number_input("Bwd Packet Length Mean", 0.0, 1e6, 500.0)
                submitted = st.form_submit_button("🔍 Analyser", width='stretch')

        if submitted:
            session = {
                "duration": duration, "fwd_packets": fwd_packets, "bwd_packets": bwd_packets,
                "fwd_len": fwd_len, "bwd_len": bwd_len,
                "flow_bytes_s": flow_bytes_s, "flow_packets_s": flow_packets_s,
                "fwd_iat_mean": fwd_iat_mean, "bwd_iat_mean": bwd_iat_mean,
                "fwd_psh_flags": fwd_psh_flags, "syn_count": syn_count,
                "rst_count": rst_count, "ack_count": ack_count, "psh_count": psh_count,
                "active_mean": active_mean, "idle_mean": idle_mean,
                "flow_iat_mean": flow_iat_mean,
                "fwd_len_mean": fwd_len_mean, "bwd_len_mean": bwd_len_mean,
                "dst_port": dst_port,
            }
            try:
                import explain
                pred, conf, factors = explain.explain_prediction(session, dataset=active_dataset)
            except Exception as e:
                st.error(f"Erreur: {e}")
                pred, conf, factors = "unknown", 0.5, []

            with col2:
                st.subheader("Resultat")
                if pred == "ATTACK":
                    st.error(f"🔴 ALERTE  -  Attaque detectee (confiance: {conf*100:.1f}%)")
                elif pred == "NORMAL":
                    st.success(f"🟢 Session normale (confiance: {conf*100:.1f}%)")
                else:
                    st.warning(f"⚠ Indetermine (confiance: {conf*100:.1f}%)")
                if factors:
                    st.subheader("Facteurs explicatifs")
                    for f in factors[:5]:
                        sens = "⬆" if f.get("sens") == "augmente" else "⬇"
                        st.write(f"  {sens} **{f['feature']}** : {f['contribution']:+.3f}")

    else:
        # NSL-KDD form
        with col1:
            st.subheader("Parametres de la session (NSL-KDD)")
            with st.form("session_form"):
                src_ip = st.text_input("IP Source", "192.168.1.105")
                dst_ip = st.text_input("IP Destination", "10.0.0.1")
                src_port = st.number_input("Port Source", 1024, 65535, 54321)
                dst_port = st.number_input("Port Destination", 1, 65535, 80)
                protocol = st.selectbox("Protocol", ["tcp", "udp", "icmp"])
                duration = st.number_input("Duree (s)", 0.0, 10000.0, 5.0)
                src_bytes = st.number_input("Bytes envoyes", 0, 100000000, 5000)
                dst_bytes = st.number_input("Bytes recus", 0, 100000000, 2000)
                serror_rate = st.slider("Taux SERR", 0.0, 1.0, 0.05)
                logged_in = st.selectbox("Logged in", [0, 1])
                count = st.slider("Count (conn host)", 0, 255, 10)
                srv_count = st.slider("Srv Count", 0, 255, 10)
                rerror_rate = st.slider("Taux RERR", 0.0, 1.0, 0.02)
                service = st.selectbox("Service", ["http", "ftp", "smtp", "dns", "ssh", "irc", "telnet", "pop3", "other"])
                flag = st.selectbox("Flag", ["SF", "S0", "S1", "S2", "S3", "REJ", "RSTO", "RSTR", "SH", "SHR"])
                hot = st.slider("Hot (anomalies)", 0, 10, 0)
                wrong_fragment = st.slider("Fragments corrompus", 0, 3, 0)
                urgent = st.slider("Paquets urgents", 0, 5, 0)
                num_compromised = st.slider("Hotes compromis", 0, 10, 0)
                diff_srv_rate = st.slider("Taux diff srv", 0.0, 1.0, 0.05)
                dst_host_count = st.slider("Dst host count", 0, 255, 50)
                submitted = st.form_submit_button("🔍 Analyser", width='stretch')

        if submitted:
            session = {
                "duration": duration, "protocol_type": protocol,
                "src_bytes": src_bytes, "dst_bytes": dst_bytes,
                "land": 1 if src_ip == dst_ip else 0,
                "wrong_fragment": wrong_fragment, "urgent": urgent,
                "hot": hot, "logged_in": logged_in,
                "num_compromised": num_compromised,
                "count": count, "srv_count": srv_count,
                "serror_rate": serror_rate, "srv_serror_rate": serror_rate,
                "rerror_rate": rerror_rate, "srv_rerror_rate": rerror_rate,
                "diff_srv_rate": diff_srv_rate,
                "dst_host_count": dst_host_count,
                "service": service, "flag": flag
            }
            try:
                import explain
                pred, conf, factors = explain.explain_prediction(session, dataset=active_dataset)
            except Exception as e:
                st.error(f"Erreur: {e}")
                pred, conf,                pred, conf, factors = "unknown", 0.5, []

            with col2:
                st.subheader("Resultat")
                if pred == "ATTACK":
                    st.error(f"🔴 ALERTE  -  Attaque detectee (confiance: {conf*100:.1f}%)")
                elif pred == "NORMAL":
                    st.success(f"🟢 Session normale (confiance: {conf*100:.1f}%)")
                else:
                    st.warning(f"⚠ Indetermine (confiance: {conf*100:.1f}%)")

                if factors:
                    st.subheader("Facteurs explicatifs")
                    for f in factors[:5]:
                        sens = "⬆" if f.get("sens") == "augmente" else "⬇"
                        st.write(f"  {sens} **{f['feature']}** : {f['contribution']:+.3f}")

                    st.subheader("Interpretation")
                    if pred == "ATTACK":
                        alert_factors = [f for f in factors if f.get("sens") == "augmente"]
                        if alert_factors:
                            feats = ", ".join([f["feature"] for f in alert_factors[:3]])
                            st.write(f"Cette session presente des anomalies sur : {feats}")
                    else:
                        st.write("Aucun facteur d'attaque significatif detecte.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4  -  Live Stream
# ══════════════════════════════════════════════════════════════════════════════
elif page == "4. Live Stream":
    st.title("Live Stream  -  Capture Reseau")
    st.caption(f"Dataset actif: {selected_dataset_label}")
    st.info("📡 Mode actuel : simulation. Pour le mode reel, lancez live_stream.py sur le serveur et renseez les params ci-dessous.")

    col_m, col_p = st.columns(2)
    with col_m:
        mode = st.selectbox("Mode de capture", ["socket", "pcap", "sniff", "simulation"])
    with col_p:
        port = st.number_input("Port (socket)", 1, 65535, 9999)

    seuil = st.slider("Seuil d'alerte (score)", 0.0, 1.0, 0.5, 0.05)
    max_sessions = st.number_input("Nombre de sessions max", 10, 1000, 100)

    if "live_running" not in st.session_state:
        st.session_state.live_running = False
    if "live_sessions" not in st.session_state:
        st.session_state.live_sessions = []
    if "live_alerts" not in st.session_state:
        st.session_state.live_alerts = 0

    col_start, col_stop = st.columns(2)
    start_btn = col_start.button("▶ Demarrer la capture", type="primary", width='stretch')
    stop_btn = col_stop.button("■ Arreter", width='stretch')

    if start_btn:
        st.session_state.live_running = True
    if stop_btn:
        st.session_state.live_running = False

    stats_col1, stats_col2 = st.columns(2)
    sc1 = stats_col1.empty()
    sc2 = stats_col2.empty()
    alert_feed = st.empty()

    if st.session_state.live_running:
        import time, random
        sc1.metric("Sessions capturees", len(st.session_state.live_sessions))
        sc2.metric("Alertes", st.session_state.live_alerts)

        def live_gen():
            if active_dataset == "cicids2017":
                return gen_session_cicids2017(len(st.session_state.live_sessions))
            else:
                protocols = ["tcp", "udp", "icmp"]
                services = ["http", "ftp", "smtp", "dns", "ssh"]
                return {
                    "session_id": f"LIVE-{len(st.session_state.live_sessions)+1:05d}",
                    "src_ip": f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
                    "dst_ip": f"10.0.{random.randint(1,254)}.{random.randint(1,254)}",
                    "src_port": random.randint(1024, 65535),
                    "dst_port": random.choice([80, 443, 22, 21, 53]),
                    "protocol": random.choice(protocols),
                    "duration": round(random.uniform(0.1, 100), 2),
                    "src_bytes": int(random.lognormvariate(8, 3)),
                    "dst_bytes": int(random.lognormvariate(10, 2)),
                    "serror_rate": round(min(random.expovariate(2), 1), 3),
                    "logged_in": random.randint(0, 1),
                    "count": random.randint(0, 100),
                    "srv_count": random.randint(0, 100),
                    "rerror_rate": round(min(random.expovariate(3), 1), 3),
                    "service": random.choice(services),
                    "flag": random.choice(["SF", "S0", "REJ", "RSTR"]),
                    "wrong_fragment": random.randint(0, 3),
                    "urgent": random.randint(0, 2),
                    "hot": random.randint(0, 10),
                    "num_compromised": random.randint(0, 8),
                    "diff_srv_rate": round(min(random.expovariate(3), 1), 3),
                    "dst_host_count": random.randint(0, 255),
                }

        session = live_gen()
        try:
            import explain
            pred, conf, _ = explain.explain_prediction(session, dataset=active_dataset)
        except:
            pred, conf = "NORMAL", 0.5

        is_alert = (pred == "ATTACK" and conf >= seuil)
        st.session_state.live_sessions.append(session)
        st.session_state.live_alerts += int(is_alert)

        if len(st.session_state.live_sessions) > max_sessions:
            st.session_state.live_sessions = st.session_state.live_sessions[-max_sessions:]

        recent = st.session_state.live_sessions[-20:]
        rows = []
        for s in reversed(recent):
            try:
                p, c, _ = explain.explain_prediction(s, dataset=active_dataset)
            except:
                p, c = "NORMAL", 0.5
            is_a = (p == "ATTACK" and c >= seuil)
            rows.append({
                "session_id": s.get("session_id", "?"),
                "src_ip": s["src_ip"],
                "protocol": s.get("protocol", "?"),
                "bytes": f"{s.get('src_bytes', s.get('fwd_len', 0))}/{s.get('dst_bytes', s.get('bwd_len', 0))}",
                "statut": "🔴 ALERTE" if is_a else "🟢 Normal",
                "conf": f"{c*100:.1f}%"
            })

        df = pd.DataFrame(rows)
        alert_feed.dataframe(df, width='stretch', hide_index=True,
                             column_config={"statut": st.column_config.TextColumn("Statut")})

        time.sleep(0.5)
        st.rerun()
    else:
        if st.session_state.live_sessions:
            sc1.metric("Sessions capturees", len(st.session_state.live_sessions))
            sc2.metric("Alertes totales", st.session_state.live_alerts)
        else:
            st.info("Cliquez sur 'Demarrer la capture' pour lancer l'analyse temps reel.")

    st.markdown("---")
    st.subheader("Documentation reseau")
    with st.expander("Comment connecter un reseau reel"):
        st.markdown("""
        ### Option 1  -  Scapy (sniff direct)
        ```bash
        # Lancer le sniff sur une interface (necessite root)
        sudo python3 -c "
        from scapy.all import sniff, IP, TCP
        def handle(p): print(p[IP].src, p[IP].dst, p[TCP].sport, p[TCP].dport)
        sniff(iface='eth0', prn=handle, count=100)
        "
        ```
        ### Option 2  -  Zeek
        ```
        # Ajouter a local.zeek:
        Log::default_writer = Log::WRITER_ASCII(Log::json_log);
        # Ou utiliser EVE-JSON de Suricata
        ```
        ### Option 3  -  Suricata (EVE-JSON)
        ```
        # /etc/suricata/suricata.yaml
        outputs:
          - eve-log:
              enabled: yes
              type: socket
              destination: localhost:9999
              protocol: tcp
        ```
        ### Format JSON attendu
        ```json
        {
          "session_id": " uniquely",
          "src_ip": "192.168.1.1",
          "dst_ip": "10.0.0.5",
          "src_port": 54321,
          "dst_port": 80,
          "protocol": "TCP",
          "bytes_sent": 5000,
          "bytes_received": 2000,
          "packets": 10,
          "duration": 1.5,
          "flags": "S0"
        }
        ```
        """)
