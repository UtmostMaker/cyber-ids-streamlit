#!/usr/bin/env python3
"""
Explicabilite du modele IDS - SHAP ou importance des features.
Support multi-dataset: NSL-KDD et CICIDS2017.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE, "artifacts")
MODELS_DIR = os.path.join(BASE, "models")

# ── Chargement paresseux (lazy loading) ──────────────────────────────────────
_model_cache = {}
_scaler_cache = {}
_le_cache = {}
_schema_cache = {}

def _load_dataset(dataset):
    if dataset in _model_cache:
        return _model_cache[dataset], _scaler_cache[dataset], _le_cache[dataset], _schema_cache[dataset]

    if dataset == "cicids2017":
        model_path = os.path.join(MODELS_DIR, "model_cicids2017.pkl")
        preproc_path = os.path.join(ARTIFACTS_DIR, "preprocessor_cicids2017.pkl")
        schema_path = os.path.join(ARTIFACTS_DIR, "schema_cicids2017.json")
    else:
        model_path = os.path.join(MODELS_DIR, "model.pkl")
        preproc_path = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
        schema_path = os.path.join(ARTIFACTS_DIR, "schema.json")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(preproc_path, "rb") as f:
        scaler = pickle.load(f)
    with open(schema_path, "r") as f:
        schema = json.load(f)

    if dataset == "cicids2017":
        label_encoders = {}
    else:
        with open(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"), "rb") as f:
            label_encoders = pickle.load(f)

    _model_cache[dataset] = model
    _scaler_cache[dataset] = scaler
    _le_cache[dataset] = label_encoders
    _schema_cache[dataset] = schema

    return model, scaler, label_encoders, schema

# ── SHAP ─────────────────────────────────────────────────────────────────────
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP disponible.")
except ImportError:
    print("SHAP non disponible. Utilisation de l'importance des features.")


def explain_prediction(session_dict, dataset="nslkdd"):
    """
    Explique une prediction pour une session donnee.

    Args:
        session_dict: dict avec les valeurs des features
        dataset: "nslkdd" ou "cicids2017"

    Returns:
        dict avec prediction, confiance, et top 3 facteurs
    """
    model, scaler, label_encoders, schema = _load_dataset(dataset)
    feature_names = schema["feature_names"]

    # ── Construction du vecteur de features ─────────────────────────────────
    if dataset == "cicids2017":
        # CICIDS2017: toutes les features sont numeriques
        vec = []
        for feat in feature_names:
            raw_val = session_dict.get(feat, 0)
            # Appliquer le preprocessor (StandardScaler)
            vec.append(float(raw_val))
        X_raw = pd.DataFrame([vec], columns=feature_names)
        X = pd.DataFrame(scaler.transform(X_raw), columns=feature_names)
    else:
        # NSL-KDD: features categorielles a encoder
        vec = []
        for feat in feature_names:
            if feat in label_encoders:
                le = label_encoders[feat]
                val = session_dict.get(feat, "unknown")
                if val not in le.classes_:
                    val = le.classes_[0]
                vec.append(le.transform([val])[0])
            else:
                vec.append(float(session_dict.get(feat, 0)))
        X = pd.DataFrame([vec], columns=feature_names)

    # ── Prediction ────────────────────────────────────────────────────────────
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confiance = float(proba[1] if pred == 1 else proba[0])

    # ── Facteurs explicatifs (feature importance) ────────────────────────────
    if dataset == "cicids2017":
        fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance_cicids2017.csv")
    else:
        fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.csv")

    fi = pd.read_csv(fi_path)
    fi_sorted = fi.sort_values("importance", ascending=False).head(5)
    facteurs = []
    for _, row in fi_sorted.iterrows():
        feat_name = row["feature"]
        feat_val = vec[feature_names.index(feat_name)] if feat_name in feature_names else 0
        facteurs.append({
            "feature": feat_name,
            "sens": "influence" if pred == 1 else "typique",
            "valeur": float(feat_val),
            "importance": float(row["importance"])
        })

    pred_label = "ATTACK" if pred == 1 else "NORMAL"

    return {
        "prediction": pred_label,
        "confiance": confiance,
        "probabilites": {"normal": float(proba[0]), "attack": float(proba[1])},
        "top_facteurs": facteurs,
        "explication": f"{pred_label} (confiance: {confiance:.1%})"
    }


def get_feature_importance(dataset="nslkdd"):
    """Retourne l'importance globale des features."""
    if dataset == "cicids2017":
        fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance_cicids2017.csv")
    else:
        fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.csv")
    fi = pd.read_csv(fi_path)
    return fi.sort_values("importance", ascending=False).to_dict("records")


def get_model_predictions(X, y, dataset="nslkdd"):
    """Retourne les predictions et probabilites pour un ensemble de donnees."""
    model, _, _, _ = _load_dataset(dataset)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return y_pred, y_proba


if __name__ == "__main__":
    # Test NSL-KDD
    print("=== Test NSL-KDD ===")
    test_session = {
        "duration": 5000, "src_bytes": 1000000, "dst_bytes": 50000,
        "protocol_type": "TCP", "service": "http", "flag": "SF",
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 1,
        "logged_in": 1, "num_compromised": 2, "count": 20, "srv_count": 15,
        "serror_rate": 0.05, "srv_serror_rate": 0.05, "rerror_rate": 0.01,
        "srv_rerror_rate": 0.01, "diff_srv_rate": 0.2, "dst_host_count": 150
    }
    result = explain_prediction(test_session, dataset="nslkdd")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Test CICIDS2017
    print("\n=== Test CICIDS2017 ===")
    test_session2 = {
        "duration": 5000, "fwd_packets": 10, "bwd_packets": 8,
        "fwd_len": 5000, "bwd_len": 3000,
        "flow_bytes_s": 1e6, "flow_packets_s": 100,
        "fwd_iat_mean": 0.5, "bwd_iat_mean": 0.5,
        "fwd_psh_flags": 1, "syn_count": 2, "rst_count": 0,
        "ack_count": 10, "psh_count": 1,
        "active_mean": 0.3, "idle_mean": 0.5,
        "flow_iat_mean": 0.5, "fwd_len_mean": 500, "bwd_len_mean": 375,
        "dst_port": 80
    }
    result2 = explain_prediction(test_session2, dataset="cicids2017")
    print(json.dumps(result2, indent=2, ensure_ascii=False))
