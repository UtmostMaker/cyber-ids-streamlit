#!/usr/bin/env python3
"""
Preparation des donnees IDS - generation du dataset et preprocessing.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
ARTIFACTS_DIR = os.path.join(BASE, "artifacts")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("=== Generation du dataset synthétique IDS ===")

n_samples = 3000
n_normal = int(n_samples * 0.75)
n_attack = n_samples - n_normal

# --- Generer chaque groupe separement pour avere des correlations realistes ---
np.random.seed(42)

# === NORMAL (label=0) ===
normal = pd.DataFrame()
normal["duration"] = np.abs(np.random.lognormal(3, 1.5, n_normal)).astype(int)
normal["src_bytes"] = np.abs(np.random.lognormal(6, 2, n_normal)).astype(int)
normal["dst_bytes"] = np.abs(np.random.lognormal(7, 2, n_normal)).astype(int)
normal["land"] = 0
normal["wrong_fragment"] = 0
normal["urgent"] = 0
normal["hot"] = np.random.randint(0, 3, n_normal)
normal["logged_in"] = np.random.choice([0, 1], n_normal, p=[0.1, 0.9])
normal["num_compromised"] = np.random.randint(0, 5, n_normal)
normal["count"] = np.random.randint(0, 100, n_normal)
normal["srv_count"] = np.random.randint(0, 80, n_normal)
normal["serror_rate"] = np.random.beta(1, 20, n_normal)
normal["srv_serror_rate"] = np.random.beta(1, 20, n_normal)
normal["rerror_rate"] = np.random.beta(1, 20, n_normal)
normal["srv_rerror_rate"] = np.random.beta(1, 20, n_normal)
normal["diff_srv_rate"] = np.random.beta(2, 5, n_normal)
normal["dst_host_count"] = np.random.randint(50, 250, n_normal)
normal["protocol_type"] = np.random.choice(["TCP", "UDP", "ICMP"], n_normal, p=[0.75, 0.20, 0.05])
normal["service"] = np.random.choice(["http", "https", "ssh", "dns", "smtp", "ftp"], n_normal, p=[0.30, 0.25, 0.15, 0.10, 0.10, 0.10])
normal["flag"] = np.random.choice(["SF", "S0", "REJ"], n_normal, p=[0.85, 0.05, 0.10])
normal["label"] = 0

# === ATTAQUE (label=1) ===
attack = pd.DataFrame()
attack["duration"] = np.abs(np.random.lognormal(1, 2, n_attack)).astype(int)
attack["src_bytes"] = np.abs(np.random.lognormal(2, 3, n_attack)).astype(int)
attack["dst_bytes"] = np.abs(np.random.lognormal(1, 2, n_attack)).astype(int)
attack["land"] = np.random.choice([0, 1], n_attack, p=[0.85, 0.15])
attack["wrong_fragment"] = np.random.randint(0, 4, n_attack)
attack["urgent"] = np.random.randint(0, 3, n_attack)
attack["hot"] = np.random.randint(0, 10, n_attack)
attack["logged_in"] = np.random.choice([0, 1], n_attack, p=[0.7, 0.3])
attack["num_compromised"] = np.random.randint(5, 20, n_attack)
attack["count"] = np.random.randint(50, 200, n_attack)
attack["srv_count"] = np.random.randint(0, 30, n_attack)
attack["serror_rate"] = np.random.beta(5, 2, n_attack)
attack["srv_serror_rate"] = np.random.beta(4, 3, n_attack)
attack["rerror_rate"] = np.random.beta(3, 4, n_attack)
attack["srv_rerror_rate"] = np.random.beta(3, 4, n_attack)
attack["diff_srv_rate"] = np.random.beta(1, 3, n_attack)
attack["dst_host_count"] = np.random.randint(0, 100, n_attack)
attack["protocol_type"] = np.random.choice(["TCP", "UDP", "ICMP"], n_attack, p=[0.40, 0.30, 0.30])
attack["service"] = np.random.choice(["other", "http", "ftp", "telnet", "smtp"], n_attack, p=[0.35, 0.20, 0.15, 0.15, 0.15])
attack["flag"] = np.random.choice(["S0", "REJ", "RSTR", "SH", "OTH", "SF"], n_attack, p=[0.35, 0.20, 0.15, 0.10, 0.10, 0.10])
attack["label"] = 1

# Fusionner et melanger
df = pd.concat([normal, attack], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Sauvegarder CSV
csv_path = os.path.join(DATA_DIR, "cybersecurity_intrusion_data.csv")
df.to_csv(csv_path, index=False)
print(f"Dataset sauvegarde : {csv_path} ({len(df)} lignes)")
print(f"  Normal : {(df.label==0).sum()} | Attaque : {(df.label==1).sum()}")

# --- Preprocessing ---
print("\n=== Preprocessing ===")
categorical_cols = ["protocol_type", "service", "flag"]
numerical_cols = [c for c in df.columns if c not in categorical_cols + ["label"]]
print(f"Categoriques : {categorical_cols}")
print(f"Numeriques   : {numerical_cols}")

label_encoders = {}
df_encoded = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

X_all = df_encoded.drop(columns=["label"])
y_all = df_encoded["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

with open(os.path.join(ARTIFACTS_DIR, "train.pkl"), "wb") as f:
    pickle.dump((X_train, y_train), f)
with open(os.path.join(ARTIFACTS_DIR, "test.pkl"), "wb") as f:
    pickle.dump((X_test, y_test), f)
with open(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"), "wb") as f:
    pickle.dump(label_encoders, f)

schema = {
    "n_samples": len(df),
    "n_features": X_all.shape[1],
    "feature_names": list(X_all.columns),
    "categorical_cols": categorical_cols,
    "numerical_cols": numerical_cols,
    "target": "label",
    "target_classes": {"0": "normal", "1": "attaque"},
    "train_size": len(X_train),
    "test_size": len(X_test),
    "class_distribution": {"normal": int((y_all==0).sum()), "attaque": int((y_all==1).sum())}
}
with open(os.path.join(ARTIFACTS_DIR, "schema.json"), "w") as f:
    json.dump(schema, f, indent=2)

print(f"\nArtefacts : train.pkl, test.pkl, label_encoders.pkl, schema.json")
print("Preprocessing termine.")
