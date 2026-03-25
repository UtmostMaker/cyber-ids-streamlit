#!/usr/bin/env python3
"""
Conversion du dataset NSL-KDD en CSV pour le projet Cyber IDS.
Source: https://github.com/defcom17/NSL_KDD
"""
import os
import json
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
ARTIFACTS_DIR = os.path.join(BASE, "artifacts")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Noms des colonnes NSL-KDD (sans header)
NSL_KDD_COLS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_rerror_rate", "dst_host_serror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty_level"
]

print("=== Chargement NSL-KDD ===")
train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
test_path = os.path.join(DATA_DIR, "KDDTest+.txt")

df_train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLS)
df_test = pd.read_csv(test_path, header=None, names=NSL_KDD_COLS)

print(f"Train: {len(df_train)} lignes | Test: {len(df_test)} lignes")

# Labels d'attaque (normal=0, tout le reste=1)
attack_labels = set(df_train["label"].unique()) - {"normal"}
print(f"Types d'attaques trouves ({len(attack_labels)}): {sorted(attack_labels)}")

df_train["binary_label"] = df_train["label"].apply(lambda x: 0 if x == "normal" else 1)
df_test["binary_label"] = df_test["label"].apply(lambda x: 0 if x == "normal" else 1)

# Fusion train + test pour avoir un seul dataset
df_all = pd.concat([df_train, df_test], ignore_index=True)
print(f"Total: {len(df_all)} lignes | Normales: {(df_all.binary_label==0).sum()} | Attaques: {(df_all.binary_label==1).sum()}")

# --- Selection des 20 features les plus pertinentes pour IDS ---
SELECTED_FEATURES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "logged_in", "num_compromised", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "diff_srv_rate", "dst_host_count"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
NUMERICAL_COLS = [f for f in SELECTED_FEATURES if f not in CATEGORICAL_COLS]

print(f"\nFeatures selectionnees ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")
print(f"Categoriques: {CATEGORICAL_COLS}")
print(f"Numeriques: {NUMERICAL_COLS}")

# Construire le DataFrame final
df = df_all[SELECTED_FEATURES + ["binary_label"]].copy()
df = df.rename(columns={"binary_label": "label"})

# --- Sous-echantillonnage pour rester sous 10MB ---
# ~10k lignes suffisent pour une demo solide
TARGET_SIZE = 10000
if len(df) > TARGET_SIZE:
    # Stratifier pour garder la proportion d'attaques
    df = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(n=max(1, int(TARGET_SIZE * len(x) / len(df))), random_state=42)
    ).reset_index(drop=True)
    print(f"\nSous-echantillonnage a {len(df)} lignes")
    print(f"  Normales: {(df.label==0).sum()} | Attaques: {(df.label==1).sum()}")

# --- Encodage des variables categoriques ---
label_encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# --- Train / Test split ---
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# --- Sauvegarder CSV ---
csv_path = os.path.join(DATA_DIR, "cybersecurity_intrusion_data.csv")
df.to_csv(csv_path, index=False)
csv_size = os.path.getsize(csv_path)
print(f"CSV sauvegarde: {csv_path} ({csv_size / 1024 / 1024:.2f} MB, {len(df)} lignes)")

# --- Artefacts pour entrainement ---
with open(os.path.join(ARTIFACTS_DIR, "train.pkl"), "wb") as f:
    pickle.dump((X_train, y_train), f)
with open(os.path.join(ARTIFACTS_DIR, "test.pkl"), "wb") as f:
    pickle.dump((X_test, y_test), f)
with open(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"), "wb") as f:
    pickle.dump(label_encoders, f)

schema = {
    "dataset": "NSL-KDD (Cyber IDS Subset)",
    "source": "https://github.com/defcom17/NSL_KDD",
    "original_train_rows": int(len(df_train)),
    "original_test_rows": int(len(df_test)),
    "n_samples": int(len(df)),
    "n_features": int(X.shape[1]),
    "feature_names": list(X.columns),
    "categorical_cols": CATEGORICAL_COLS,
    "numerical_cols": NUMERICAL_COLS,
    "target": "label",
    "target_classes": {"0": "normal", "1": "attaque"},
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "class_distribution": {"normal": int((y==0).sum()), "attaque": int((y==1).sum())}
}
with open(os.path.join(ARTIFACTS_DIR, "schema.json"), "w") as f:
    json.dump(schema, f, indent=2)

print("\n=== Artefacts sauvegardes ===")
print(f"  {csv_path} ({csv_size / 1024 / 1024:.2f} MB)")
print(f"  {ARTIFACTS_DIR}/train.pkl")
print(f"  {ARTIFACTS_DIR}/test.pkl")
print(f"  {ARTIFACTS_DIR}/label_encoders.pkl")
print(f"  {ARTIFACTS_DIR}/schema.json")
print("\nPreprocessing NSL-KDD termine.")
