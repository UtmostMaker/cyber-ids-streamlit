#!/usr/bin/env python3
"""
Preparation des donnees CICIDS2017 pour le projet IDS.

Le dataset CICIDS2017 original contient des flux reseau captures avec CICFlowMeter.
Ce script genere un dataset synthetique base sur les statistiques reelles rapportees
dans la litterature scientifique pour CICIDS2017, avec des distributions plus
realistes et plus complexes que NSL-KDD.

Reference: Sharafaldin et al., "Toward Generating a New Intrusion Detection
Dataset and Intrusion Traffic Characterization", ICISSP 2018.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
ARTIFACTS_DIR = os.path.join(BASE, "artifacts")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

np.random.seed(42)

print("=== Generation du dataset synthetique CICIDS2017 ===")
print("Base sur les statistiques reelles de CICIDS2017 (Sharafaldin et al., 2018)")

n_samples = 15000
n_normal = int(n_samples * 0.60)
n_attack = n_samples - n_normal

# ============================================================
# Features CICIDS2017 disponibles (20 features selectionnees)
# Mapping vers les noms internes du projet
# ============================================================
FEATURE_NAMES = [
    "duration",          # 1. Flow Duration
    "fwd_packets",       # 2. Total Fwd Packets
    "bwd_packets",       # 3. Total Backward Packets
    "fwd_len",           # 4. Total Length of Fwd Packets
    "bwd_len",           # 5. Total Length of Bwd Packets
    "flow_bytes_s",      # 6. Flow Bytes/s
    "flow_packets_s",    # 7. Flow Packets/s
    "fwd_iat_mean",      # 8. Fwd IAT Mean
    "bwd_iat_mean",      # 9. Bwd IAT Mean
    "fwd_psh_flags",     # 10. Fwd PSH Flags Count
    "syn_count",         # 11. SYN Flag Count
    "rst_count",         # 12. RST Flag Count
    "ack_count",         # 13. ACK Flag Count
    "psh_count",         # 14. PSH Flag Count
    "active_mean",       # 15. Active Mean
    "idle_mean",         # 16. Idle Mean
    "flow_iat_mean",     # 17. Flow IAT Mean
    "fwd_len_mean",      # 18. Fwd Packet Length Mean
    "bwd_len_mean",      # 19. Bwd Packet Length Mean
    "dst_port",          # 20. Destination Port
]

# ============================================================
# Types d'attaques CICIDS2017 simulees
# ============================================================
ATTACK_TYPES = {
    "Benign": {"weight": 0.60, "label": 0},
    "BruteForce": {"weight": 0.10, "label": 1},
    "PortScan": {"weight": 0.10, "label": 1},
    "DoS": {"weight": 0.15, "label": 1},
    "Infiltration": {"weight": 0.05, "label": 1},
}

def sample_attack_type(weights_dict, n):
    """Echantillonne les types d'attaques selon les poids."""
    names = list(weights_dict.keys())
    probs = [weights_dict[n]["weight"] for n in names]
    return np.random.choice(names, n, p=probs)

# ============================================================
# Distribution normale (BENIGN)
# Sessions reseau normales avec caracteristiques realistes
# ============================================================
print("\n--- Generation du trafic normal ---")
n_benign = n_normal
n_per_attack = {
    "BruteForce": int(n_attack * 0.20),
    "PortScan": int(n_attack * 0.25),
    "DoS": int(n_attack * 0.40),
    "Infiltration": n_attack - int(n_attack * 0.20) - int(n_attack * 0.25) - int(n_attack * 0.40),
}

def gen_benign(n):
    df = pd.DataFrame()
    # Distributions melangees pour simuler le bruit et la variabilite du trafic reel
    # 60% du trafic est vraiment normal, 40% ressemble a du trafic mixte
    mix = np.random.random(n)
    df["duration"] = np.where(
        mix < 0.6,
        np.abs(np.random.lognormal(5, 1.5, n)),
        np.abs(np.random.lognormal(3, 2, n))
    ).astype(int)
    df["fwd_packets"] = np.where(
        mix < 0.6,
        np.random.poisson(10, n),
        np.random.poisson(5, n)
    )
    df["bwd_packets"] = np.where(
        mix < 0.6,
        np.random.poisson(8, n),
        np.random.poisson(3, n)
    )
    df["fwd_len"] = np.where(
        mix < 0.6,
        np.abs(np.random.lognormal(10, 1.5, n)),
        np.abs(np.random.lognormal(5, 2, n))
    ).astype(int)
    df["bwd_len"] = np.where(
        mix < 0.6,
        np.abs(np.random.lognormal(11, 1.5, n)),
        np.abs(np.random.lognormal(6, 2, n))
    ).astype(int)
    df["flow_bytes_s"] = np.where(
        mix < 0.6,
        np.abs(np.random.lognormal(13, 1.5, n)),
        np.abs(np.random.lognormal(8, 2, n))
    )
    df["flow_packets_s"] = np.where(
        mix < 0.6,
        np.abs(np.random.lognormal(3.5, 1.2, n)),
        np.abs(np.random.lognormal(2, 1.5, n))
    )
    df["fwd_iat_mean"] = np.where(
        mix < 0.6,
        np.abs(np.random.lognormal(5.5, 1.5, n)),
        np.abs(np.random.lognormal(3, 2, n))
    )
    df["bwd_iat_mean"] = np.where(
        mix < 0.6,
        np.abs(np.random.lognormal(6.5, 1.5, n)),
        np.abs(np.random.lognormal(4, 2, n))
    )
    df["fwd_psh_flags"] = np.random.poisson(1, n)
    df["syn_count"] = np.where(
        mix < 0.6,
        np.random.poisson(2, n),
        np.random.poisson(5, n)
    )
    df["rst_count"] = np.random.poisson(0.5, n)
    df["ack_count"] = np.where(
        mix < 0.6,
        np.abs(np.random.lognormal(4, 1, n)).astype(int),
        np.abs(np.random.lognormal(3, 1.5, n)).astype(int)
    )
    df["psh_count"] = np.random.poisson(1, n)
    df["active_mean"] = np.where(
        mix < 0.6,
        np.abs(np.random.lognormal(8, 1.5, n)),
        np.abs(np.random.lognormal(4, 2, n))
    )
    df["idle_mean"] = np.where(
        mix < 0.6,
        np.abs(np.random.lognormal(9, 1.5, n)),
        np.abs(np.random.lognormal(5, 2, n))
    )
    df["flow_iat_mean"] = np.where(
        mix < 0.6,
        np.abs(np.random.lognormal(6, 1.5, n)),
        np.abs(np.random.lognormal(3.5, 2, n))
    )
    df["fwd_len_mean"] = np.where(
        mix < 0.6,
        np.abs(np.random.lognormal(9, 1.5, n)),
        np.abs(np.random.lognormal(5, 2, n))
    )
    df["bwd_len_mean"] = np.where(
        mix < 0.6,
        np.abs(np.random.lognormal(10, 1.5, n)),
        np.abs(np.random.lognormal(6, 2, n))
    )
    df["dst_port"] = np.random.choice([80, 443, 22, 53, 8080, 21, 25], n,
                                       p=[0.30, 0.25, 0.15, 0.10, 0.08, 0.07, 0.05])
    df["label"] = 0
    df["attack_type"] = "Benign"
    return df

# ============================================================
# Brute Force SSH/FTP
# Nombreuses connexions au port 22/21, courte duree, flags SYN+ACK
# ============================================================
def gen_bruteforce(n):
    # 70% agressif (evident), 30% stealth (similar to normal)
    stealth = np.random.random(n) > 0.7
    df = pd.DataFrame()
    df["duration"] = np.where(
        stealth,
        np.abs(np.random.lognormal(4.5, 1, n)),
        np.abs(np.random.lognormal(1, 1.5, n))
    ).astype(int)
    df["fwd_packets"] = np.where(stealth, np.random.poisson(8, n), np.random.poisson(3, n))
    df["bwd_packets"] = np.where(stealth, np.random.poisson(6, n), np.random.poisson(2, n))
    df["fwd_len"] = np.where(stealth, np.abs(np.random.lognormal(9, 1.5, n)).astype(int), np.abs(np.random.lognormal(4, 2, n)).astype(int))
    df["bwd_len"] = np.where(stealth, np.abs(np.random.lognormal(10, 1.5, n)).astype(int), np.abs(np.random.lognormal(4, 2, n)).astype(int))
    df["flow_bytes_s"] = np.where(stealth, np.abs(np.random.lognormal(12, 1.5, n)), np.abs(np.random.lognormal(8, 2, n)))
    df["flow_packets_s"] = np.where(stealth, np.abs(np.random.lognormal(3, 1.5, n)), np.abs(np.random.lognormal(5, 2, n)))
    df["fwd_iat_mean"] = np.where(stealth, np.abs(np.random.lognormal(5, 1.5, n)), np.abs(np.random.lognormal(2, 1.5, n)))
    df["bwd_iat_mean"] = np.where(stealth, np.abs(np.random.lognormal(6, 1.5, n)), np.abs(np.random.lognormal(2, 1.5, n)))
    df["fwd_psh_flags"] = np.where(stealth, np.random.poisson(1, n), np.random.poisson(2, n))
    df["syn_count"] = np.where(stealth, np.random.poisson(2, n), np.random.poisson(8, n))
    df["rst_count"] = np.where(stealth, np.random.poisson(0, n), np.random.poisson(1, n))
    df["ack_count"] = np.where(stealth, np.abs(np.random.lognormal(4, 1, n)).astype(int), np.abs(np.random.lognormal(2, 1, n)).astype(int))
    df["psh_count"] = np.where(stealth, np.random.poisson(1, n), np.random.poisson(0, n))
    df["active_mean"] = np.where(stealth, np.abs(np.random.lognormal(7, 1.5, n)), np.abs(np.random.lognormal(1, 1, n)))
    df["idle_mean"] = np.where(stealth, np.abs(np.random.lognormal(8, 1.5, n)), np.abs(np.random.lognormal(0.5, 0.5, n)))
    df["flow_iat_mean"] = np.where(stealth, np.abs(np.random.lognormal(5.5, 1.5, n)), np.abs(np.random.lognormal(1.5, 1, n)))
    df["fwd_len_mean"] = np.where(stealth, np.abs(np.random.lognormal(8, 1.5, n)), np.abs(np.random.lognormal(3, 1.5, n)))
    df["bwd_len_mean"] = np.where(stealth, np.abs(np.random.lognormal(9, 1.5, n)), np.abs(np.random.lognormal(3, 1.5, n)))
    df["dst_port"] = np.where(stealth, np.random.choice([22, 21, 23], n, p=[0.7, 0.2, 0.1]), np.random.choice([22, 21, 23], n, p=[0.7, 0.2, 0.1]))
    df["label"] = 1
    df["attack_type"] = "BruteForce"
    return df

# ============================================================
# Port Scan
# Nombreuses connexions, plusieurs ports dest differents, peu de bytes
# ============================================================
def gen_portscan(n):
    # 30% obvious (classic port scan), 70% stealth (slow/progressive scan)
    stealth = np.random.random(n) > 0.3
    df = pd.DataFrame()
    df["duration"] = np.where(
        stealth,
        np.abs(np.random.lognormal(5, 1.5, n)),
        np.abs(np.random.lognormal(2, 2, n))
    ).astype(int)
    df["fwd_packets"] = np.where(stealth, np.random.poisson(5, n), np.random.poisson(1, n))
    df["bwd_packets"] = np.where(stealth, np.random.poisson(3, n), np.random.poisson(0, n))
    df["fwd_len"] = np.where(stealth, np.abs(np.random.lognormal(8, 2, n)).astype(int), np.abs(np.random.lognormal(2, 1, n)).astype(int))
    df["bwd_len"] = np.where(stealth, np.abs(np.random.lognormal(7, 2, n)).astype(int), np.abs(np.random.lognormal(0, 0.1, n)).astype(int))
    df["flow_bytes_s"] = np.where(stealth, np.abs(np.random.lognormal(10, 2, n)), np.abs(np.random.lognormal(2, 1.5, n)))
    df["flow_packets_s"] = np.where(stealth, np.abs(np.random.lognormal(2, 1.5, n)), np.abs(np.random.lognormal(1, 1, n)))
    df["fwd_iat_mean"] = np.where(stealth, np.abs(np.random.lognormal(4, 2, n)), np.abs(np.random.lognormal(1, 1, n)))
    df["bwd_iat_mean"] = np.where(stealth, np.abs(np.random.lognormal(5, 2, n)), np.abs(np.random.lognormal(0.1, 0.1, n)))
    df["fwd_psh_flags"] = np.random.poisson(0, n)
    df["syn_count"] = np.where(stealth, np.random.poisson(2, n), np.random.poisson(1, n))
    df["rst_count"] = np.random.poisson(0, n)
    df["ack_count"] = np.where(stealth, np.abs(np.random.lognormal(3, 1, n)).astype(int), np.abs(np.random.lognormal(0, 0.1, n)).astype(int))
    df["psh_count"] = np.random.poisson(0, n)
    df["active_mean"] = np.where(stealth, np.abs(np.random.lognormal(5, 2, n)), np.abs(np.random.lognormal(0.1, 0.1, n)))
    df["idle_mean"] = np.where(stealth, np.abs(np.random.lognormal(6, 2, n)), np.abs(np.random.lognormal(0.1, 0.1, n)))
    df["flow_iat_mean"] = np.where(stealth, np.abs(np.random.lognormal(4, 2, n)), np.abs(np.random.lognormal(1, 1, n)))
    df["fwd_len_mean"] = np.where(stealth, np.abs(np.random.lognormal(7, 2, n)), np.abs(np.random.lognormal(2, 1, n)))
    df["bwd_len_mean"] = np.where(stealth, np.abs(np.random.lognormal(6, 2, n)), np.abs(np.random.lognormal(0, 0.1, n)))
    df["dst_port"] = np.random.randint(1, 65535, n)
    df["label"] = 1
    df["attack_type"] = "PortScan"
    return df

# ============================================================
# DoS (SYN Flood, UDP Flood, etc.)
# Debit eleve, banyak packets, courte duree
# ============================================================
def gen_dos(n):
    # 40% obvious flood, 60% low-rate/slow DoS that mimics normal
    stealth = np.random.random(n) > 0.4
    df = pd.DataFrame()
    df["duration"] = np.where(stealth, np.abs(np.random.lognormal(4, 2, n)).astype(int), np.abs(np.random.lognormal(1.5, 1.5, n)).astype(int))
    df["fwd_packets"] = np.where(stealth, np.random.poisson(15, n), np.random.poisson(100, n))
    df["bwd_packets"] = np.where(stealth, np.random.poisson(5, n), np.random.poisson(5, n))
    df["fwd_len"] = np.where(stealth, np.abs(np.random.lognormal(9, 2, n)).astype(int), np.abs(np.random.lognormal(6, 3, n)).astype(int))
    df["bwd_len"] = np.where(stealth, np.abs(np.random.lognormal(8, 2, n)).astype(int), np.abs(np.random.lognormal(2, 1, n)).astype(int))
    df["flow_bytes_s"] = np.where(stealth, np.abs(np.random.lognormal(12, 2, n)), np.abs(np.random.lognormal(15, 3, n)))
    df["flow_packets_s"] = np.where(stealth, np.abs(np.random.lognormal(4, 1.5, n)), np.abs(np.random.lognormal(8, 2, n)))
    df["fwd_iat_mean"] = np.where(stealth, np.abs(np.random.lognormal(3, 2, n)), np.abs(np.random.lognormal(0.5, 0.5, n)))
    df["bwd_iat_mean"] = np.where(stealth, np.abs(np.random.lognormal(4, 2, n)), np.abs(np.random.lognormal(2, 1, n)))
    df["fwd_psh_flags"] = np.where(stealth, np.random.poisson(2, n), np.random.poisson(10, n))
    df["syn_count"] = np.where(stealth, np.random.poisson(5, n), np.random.poisson(50, n))
    df["rst_count"] = np.where(stealth, np.random.poisson(1, n), np.random.poisson(5, n))
    df["ack_count"] = np.where(stealth, np.abs(np.random.lognormal(4, 1, n)).astype(int), np.abs(np.random.lognormal(5, 2, n)).astype(int))
    df["psh_count"] = np.where(stealth, np.random.poisson(2, n), np.random.poisson(10, n))
    df["active_mean"] = np.where(stealth, np.abs(np.random.lognormal(5, 2, n)), np.abs(np.random.lognormal(0.5, 0.5, n)))
    df["idle_mean"] = np.where(stealth, np.abs(np.random.lognormal(6, 2, n)), np.abs(np.random.lognormal(0.1, 0.1, n)))
    df["flow_iat_mean"] = np.where(stealth, np.abs(np.random.lognormal(3.5, 2, n)), np.abs(np.random.lognormal(0.5, 0.5, n)))
    df["fwd_len_mean"] = np.where(stealth, np.abs(np.random.lognormal(8, 2, n)), np.abs(np.random.lognormal(5, 2, n)))
    df["bwd_len_mean"] = np.where(stealth, np.abs(np.random.lognormal(7, 2, n)), np.abs(np.random.lognormal(1, 1, n)))
    df["dst_port"] = np.where(stealth, np.random.choice([80, 443, 53], n, p=[0.4, 0.3, 0.3]), np.random.choice([80, 443, 53, 0], n, p=[0.4, 0.2, 0.2, 0.2]))
    df["label"] = 1
    df["attack_type"] = "DoS"
    return df

# ============================================================
# Infiltration (slow attacks, long duration, low profile)
# ============================================================
def gen_infiltration(n):
    df = pd.DataFrame()
    df["duration"] = np.abs(np.random.lognormal(8, 3, n)).astype(int)
    df["fwd_packets"] = np.random.poisson(10, n)
    df["bwd_packets"] = np.random.poisson(8, n)
    df["fwd_len"] = np.abs(np.random.lognormal(6, 3, n)).astype(int)
    df["bwd_len"] = np.abs(np.random.lognormal(6, 3, n)).astype(int)
    df["flow_bytes_s"] = np.abs(np.random.lognormal(5, 2, n))
    df["flow_packets_s"] = np.abs(np.random.lognormal(1, 1, n))
    df["fwd_iat_mean"] = np.abs(np.random.lognormal(7, 3, n))
    df["bwd_iat_mean"] = np.abs(np.random.lognormal(8, 3, n))
    df["fwd_psh_flags"] = np.random.poisson(1, n)
    df["syn_count"] = np.random.poisson(2, n)
    df["rst_count"] = np.random.poisson(0, n)
    df["ack_count"] = np.abs(np.random.lognormal(3, 1, n)).astype(int)
    df["psh_count"] = np.random.poisson(1, n)
    df["active_mean"] = np.abs(np.random.lognormal(10, 3, n))
    df["idle_mean"] = np.abs(np.random.lognormal(5, 2, n))
    df["flow_iat_mean"] = np.abs(np.random.lognormal(7, 3, n))
    df["fwd_len_mean"] = np.abs(np.random.lognormal(5, 2, n))
    df["bwd_len_mean"] = np.abs(np.random.lognormal(5, 2, n))
    df["dst_port"] = np.random.choice([80, 443, 22, 8080], n, p=[0.3, 0.3, 0.2, 0.2])
    df["label"] = 1
    df["attack_type"] = "Infiltration"
    return df

# ============================================================
# Generation et fusion
# ============================================================
print(f"Benign: {n_benign} | BruteForce: {n_per_attack['BruteForce']} | "
      f"PortScan: {n_per_attack['PortScan']} | DoS: {n_per_attack['DoS']} | "
      f"Infiltration: {n_per_attack['Infiltration']}")

benign_df = gen_benign(n_benign)
bruteforce_df = gen_bruteforce(n_per_attack["BruteForce"])
portscan_df = gen_portscan(n_per_attack["PortScan"])
dos_df = gen_dos(n_per_attack["DoS"])
infiltration_df = gen_infiltration(n_per_attack["Infiltration"])

df = pd.concat([benign_df, bruteforce_df, portscan_df, dos_df, infiltration_df],
               ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Remplacer les valeurs infinies et negatives
df = df.replace([np.inf, -np.inf], np.nan)
for col in df.select_dtypes(include=[np.floating]).columns:
    df[col] = df[col].fillna(df[col].median())

# Colonnes numeriques entieres
for col in ["duration", "fwd_packets", "bwd_packets", "fwd_len", "bwd_len",
            "ack_count", "dst_port"]:
    df[col] = df[col].astype(int)

# Colonnes numeriques continues
for col in ["flow_bytes_s", "flow_packets_s", "fwd_iat_mean", "bwd_iat_mean",
            "fwd_psh_flags", "syn_count", "rst_count", "psh_count",
            "active_mean", "idle_mean", "flow_iat_mean", "fwd_len_mean", "bwd_len_mean"]:
    df[col] = df[col].astype(float)

print(f"\nDataset genere: {len(df)} lignes")
print(f"  Benign: {(df.label==0).sum()} | Attaque: {(df.label==1).sum()}")
print(f"  Distribution: {df['attack_type'].value_counts().to_dict()}")

# ============================================================
# Train/Test split
# ============================================================
FEATURE_COLS = [c for c in FEATURE_NAMES if c in df.columns]
X = df[FEATURE_COLS].copy()
y = df["label"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# Sauvegarde des donnees brutes
# ============================================================
csv_path = os.path.join(DATA_DIR, "cicids2017_processed.csv")
df.to_csv(csv_path, index=False)
print(f"CSV sauvegarde: {csv_path}")

# ============================================================
# Sauvegarde des artefacts de preprocessing
# ============================================================
# StandardScaler pour les features numeriques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open(os.path.join(ARTIFACTS_DIR, "train_cicids2017.pkl"), "wb") as f:
    pickle.dump((pd.DataFrame(X_train_scaled, columns=FEATURE_COLS), y_train.reset_index(drop=True)), f)
with open(os.path.join(ARTIFACTS_DIR, "test_cicids2017.pkl"), "wb") as f:
    pickle.dump((pd.DataFrame(X_test_scaled, columns=FEATURE_COLS), y_test.reset_index(drop=True)), f)
with open(os.path.join(ARTIFACTS_DIR, "preprocessor_cicids2017.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Schema
schema = {
    "dataset": "CICIDS2017 (synthetique, base sur Sharafaldin et al. 2018)",
    "source": "HuggingFace: Lama3311/CICIDS2017 (non telechargeable, genere synthetique)",
    "n_samples": len(df),
    "n_features": len(FEATURE_COLS),
    "feature_names": FEATURE_COLS,
    "feature_names_cicids2017": FEATURE_NAMES,
    "target": "label",
    "target_classes": {"0": "Benign", "1": "Attack"},
    "attack_types": list(df["attack_type"].unique()),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "class_distribution": {
        "Benign": int((df.label==0).sum()),
        "Attack": int((df.label==1).sum()),
        "by_type": df["attack_type"].value_counts().to_dict()
    }
}
with open(os.path.join(ARTIFACTS_DIR, "schema_cicids2017.json"), "w") as f:
    json.dump(schema, f, indent=2)

print(f"\nArtefacts CICIDS2017 sauvegardes:")
print(f"  train_cicids2017.pkl | test_cicids2017.pkl | preprocessor_cicids2017.pkl")
print(f"  schema_cicids2017.json")
print("Preprocessing CICIDS2017 termine.")
