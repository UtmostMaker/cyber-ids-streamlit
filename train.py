#!/usr/bin/env python3
"""
Entrainement des modeles IDS - Random Forest et XGBoost.
Support multi-dataset: NSL-KDD (defaut) et CICIDS2017.
"""
import os
import sys
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
import argparse

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE, "artifacts")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Entrainement modeles IDS")
parser.add_argument(
    "--dataset",
    type=str,
    default="nslkdd",
    choices=["nslkdd", "cicids2017"],
    help="Dataset a utiliser (nslkdd par defaut)"
)
args = parser.parse_args()

DATASET = args.dataset
print(f"=== Dataset: {DATASET.upper()} ===\n")

# --- Chargement des donnees selon le dataset ---
if DATASET == "cicids2017":
    print("=== Chargement CICIDS2017 ===")
    train_path = os.path.join(ARTIFACTS_DIR, "train_cicids2017.pkl")
    test_path = os.path.join(ARTIFACTS_DIR, "test_cicids2017.pkl")
    preproc_path = os.path.join(ARTIFACTS_DIR, "preprocessor_cicids2017.pkl")
    results_path = os.path.join(ARTIFACTS_DIR, "results_cicids2017.json")
    model_suffix = "_cicids2017.pkl"
    fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance_cicids2017.csv")

    with open(train_path, "rb") as f:
        X_train, y_train = pickle.load(f)
    with open(test_path, "rb") as f:
        X_test, y_test = pickle.load(f)
    with open(preproc_path, "rb") as f:
        scaler = pickle.load(f)

    # Label encoders vides pour CICIDS2017 (pas de colonnes categoriques)
    label_encoders = {}
else:
    print("=== Chargement NSL-KDD ===")
    train_path = os.path.join(ARTIFACTS_DIR, "train.pkl")
    test_path = os.path.join(ARTIFACTS_DIR, "test.pkl")
    results_path = os.path.join(ARTIFACTS_DIR, "results.json")
    model_suffix = ".pkl"
    fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.csv")

    with open(train_path, "rb") as f:
        X_train, y_train = pickle.load(f)
    with open(test_path, "rb") as f:
        X_test, y_test = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"), "rb") as f:
        label_encoders = pickle.load(f)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Label distribution (train): normal={int((y_train==0).sum())} | attack={int((y_train==1).sum())}")

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
    "precision": float(f1_score(y_test, y_pred_rf, average='macro')),
    "recall": float(f1_score(y_test, y_pred_rf, average='macro')),
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
    "precision": float(f1_score(y_test, y_pred_xgb, average='macro')),
    "recall": float(f1_score(y_test, y_pred_xgb, average='macro')),
    "confusion_matrix": confusion_matrix(y_test, y_pred_xgb).tolist()
}
print(f"Accuracy : {metrics_xgb['accuracy']:.4f}")
print(f"F1       : {metrics_xgb['f1']:.4f}")
print(f"AUC-ROC  : {metrics_xgb['auc_roc']:.4f}")

# --- Comparaison ---
print("\n=== Comparaison des modeles ===")
results = {
    "dataset": DATASET,
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
model_path = os.path.join(MODELS_DIR, f"model{model_suffix}")
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
print(f"\nMeilleur modele sauvegarde : {model_path}")

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Metriques sauvegardees : {results_path}")

# Feature importance
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

fi.to_csv(fi_path, index=False)
print(f"Feature importance sauvegardee : {fi_path}")

# Confusion matrix detaillee
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred_xgb, target_names=["Normal", "Attack"]))

print(f"\n=== Entrainement {DATASET} termine avec succes ===")
