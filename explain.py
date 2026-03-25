#!/usr/bin/env python3
"""
Explicabilite du modele IDS - SHAP ou importance des features.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE, "artifacts")
MODELS_DIR = os.path.join(BASE, "models")
SCHEMA_PATH = os.path.join(ARTIFACTS_DIR, "schema.json")

# Charger modele et artefacts
with open(os.path.join(MODELS_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)
with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)

feature_names = schema["feature_names"]

# --- SHAP (avec fallback) ---
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP disponible.")
except ImportError:
    print("SHAP non disponible. Utilisation de l'importance des features.")

if SHAP_AVAILABLE:
    # Calculer les valeurs SHAP sur un echantillon
    with open(os.path.join(ARTIFACTS_DIR, "train.pkl"), "rb") as f:
        X_train, _ = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    sample = X_train.sample(n=min(200, len(X_train)), random_state=42)
    shap_values = explainer.shap_values(sample)
    # Shape: (n_samples, n_features, n_classes) pour classification
    sv = np.array(shap_values)
    if sv.ndim == 3 and sv.shape[2] == 2:
        sv_attack = sv[:, :, 1]  # classe 1 = attaque
    else:
        sv_attack = sv
    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": np.abs(sv_attack).mean(axis=0)
    }).sort_values("importance", ascending=False)
    shap_importance.to_csv(os.path.join(ARTIFACTS_DIR, "shap_importance.csv"), index=False)
    print("Importance SHAP sauvegardee.")

def explain_prediction(session_dict):
    """
    Explique une prediction pour une session donnee.
    
    Args:
        session_dict: dict avec les valeurs des features
                     (protocol_type, service, flag en texte,
                      les autres en numerique)
    
    Returns:
        dict avec prediction, confiance, et top 3 facteurs
    """
    # Construire le vecteur de features
    vec = []
    for feat in feature_names:
        if feat in ["protocol_type", "service", "flag"]:
            # Encoder avec le LabelEncoder
            le = label_encoders[feat]
            val = session_dict.get(feat, "unknown")
            if val not in le.classes_:
                val = le.classes_[0]  # fallback
            vec.append(le.transform([val])[0])
        else:
            vec.append(float(session_dict.get(feat, 0)))
    
    X = pd.DataFrame([vec], columns=feature_names)
    
    # Prediction
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confiance = float(proba[1] if pred == 1 else proba[0])
    
    # Top 3 facteurs explicatifs
    if SHAP_AVAILABLE:
        sv_local = explainer.shap_values(X)
        sv_local = np.array(sv_local)
        if sv_local.ndim == 3 and sv_local.shape[2] == 2:
            sv_local = sv_local[:, :, 1][0]  # classe 1, premier element
        elif sv_local.ndim == 2:
            sv_local = sv_local[0]
        # Prendre les 3 plus grandes valeurs absolues
        contributions = pd.DataFrame({
            "feature": feature_names,
            "contribution": list(sv_local)
        }).sort_values("contribution", key=abs, ascending=False)
        top3 = contributions.head(3)
        facteurs = []
        for _, row in top3.iterrows():
            sens = "augmente" if row["contribution"] > 0 else "diminue"
            facteurs.append({
                "feature": row["feature"],
                "sens": sens,
                "valeur": float(vec[feature_names.index(row["feature"])]),
                "contribution": float(row["contribution"])
            })
    else:
        # Fallback: importance des features
        fi = pd.read_csv(os.path.join(ARTIFACTS_DIR, "feature_importance.csv"))
        fi_sorted = fi.sort_values("importance", ascending=False).head(3)
        facteurs = []
        for _, row in fi_sorted.iterrows():
            idx = feature_names.index(row["feature"])
            val = vec[idx]
            facteurs.append({
                "feature": row["feature"],
                "sens": "influence" if pred == 1 else "typique",
                "valeur": float(val),
                "importance": float(row["importance"])
            })
    
    # Texte explicatif en francais
    if pred == 1:
        texte = f"ATTAQUE suspectee (confiance: {confiance:.1%}). "
        texte += "Facteurs cles: "
        texte += ", ".join([f"{f['feature']} ({f['sens']} le risque)" for f in facteurs])
    else:
        texte = f"SESSION NORMALE (confiance: {confiance:.1%}). "
        texte += "Fonctionnement standard. "
        texte += "Points rassurants: "
        texte += ", ".join([f["feature"] for f in facteurs])
    
    return {
        "prediction": "attaque" if pred == 1 else "normal",
        "confiance": confiance,
        "probabilites": {"normal": float(proba[0]), "attaque": float(proba[1])},
        "top_facteurs": facteurs,
        "explication": texte
    }

def get_feature_importance():
    """Retourne l'importance globale des features."""
    fi = pd.read_csv(os.path.join(ARTIFACTS_DIR, "feature_importance.csv"))
    return fi.sort_values("importance", ascending=False).to_dict("records")

if __name__ == "__main__":
    # Test rapide
    test_session = {
        "duration": 5000, "src_bytes": 1000000, "dst_bytes": 50000,
        "protocol_type": "TCP", "service": "http", "flag": "SF",
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 1,
        "logged_in": 1, "num_compromised": 2, "count": 20, "srv_count": 15,
        "serror_rate": 0.05, "srv_serror_rate": 0.05, "rerror_rate": 0.01,
        "srv_rerror_rate": 0.01, "diff_srv_rate": 0.2, "dst_host_count": 150
    }
    result = explain_prediction(test_session)
    print(json.dumps(result, indent=2, ensure_ascii=False))
