#!/usr/bin/env python3
"""
Entrainement des modeles IDS - Random Forest et XGBoost.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import xgboost as xgb

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE, "artifacts")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("=== Chargement des donnees ===")
with open(os.path.join(ARTIFACTS_DIR, "train.pkl"), "rb") as f:
    X_train, y_train = pickle.load(f)
with open(os.path.join(ARTIFACTS_DIR, "test.pkl"), "rb") as f:
    X_test, y_test = pickle.load(f)
with open(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# --- Random Forest ---
print("\n=== Entrainement Random Forest ===")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

metrics_rf = {
    "accuracy": float(accuracy_score(y_test, y_pred_rf)),
    "f1": float(f1_score(y_test, y_pred_rf)),
    "auc_roc": float(roc_auc_score(y_test, y_proba_rf)),
    "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist()
}
print(f"Accuracy : {metrics_rf['accuracy']:.4f}")
print(f"F1       : {metrics_rf['f1']:.4f}")
print(f"AUC-ROC  : {metrics_rf['auc_roc']:.4f}")

# --- XGBoost ---
print("\n=== Entrainement XGBoost ===")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

metrics_xgb = {
    "accuracy": float(accuracy_score(y_test, y_pred_xgb)),
    "f1": float(f1_score(y_test, y_pred_xgb)),
    "auc_roc": float(roc_auc_score(y_test, y_proba_xgb)),
    "confusion_matrix": confusion_matrix(y_test, y_pred_xgb).tolist()
}
print(f"Accuracy : {metrics_xgb['accuracy']:.4f}")
print(f"F1       : {metrics_xgb['f1']:.4f}")
print(f"AUC-ROC  : {metrics_xgb['auc_roc']:.4f}")

# --- Comparaison ---
print("\n=== Comparaison des modeles ===")
results = {
    "random_forest": metrics_rf,
    "xgboost": metrics_xgb,
    "best_model": None,
    "best_f1": 0.0
}

if metrics_rf["f1"] >= metrics_xgb["f1"]:
    best_name = "random_forest"
    best_metrics = metrics_rf
    best_model = rf
else:
    best_name = "xgboost"
    best_metrics = metrics_xgb
    best_model = xgb_model

results["best_model"] = best_name
results["best_f1"] = best_metrics["f1"]

print(f"Meilleur modele : {best_name}")
print(f"  Accuracy : {best_metrics['accuracy']:.4f}")
print(f"  F1       : {best_metrics['f1']:.4f}")
print(f"  AUC-ROC  : {best_metrics['auc_roc']:.4f}")

# --- Sauvegarde ---
# Meilleur modele
with open(os.path.join(MODELS_DIR, "model.pkl"), "wb") as f:
    pickle.dump(best_model, f)
print(f"\nMeilleur modele sauvegarde : {MODELS_DIR}/model.pkl")

# Metriques completes
with open(os.path.join(ARTIFACTS_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Metriques sauvegardees : {ARTIFACTS_DIR}/results.json")

# Feature importance du meilleur modele
if best_name == "random_forest":
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
else:
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)

fi.to_csv(os.path.join(ARTIFACTS_DIR, "feature_importance.csv"), index=False)
print(f"Feature importance sauvegardee : {ARTIFACTS_DIR}/feature_importance.csv")

print("\n=== Entrainement termine avec succes ===")
