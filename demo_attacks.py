#!/usr/bin/env python3
"""
Script de demonstration d'attaques simulees.
Genere des sessions reseau correspondant a des attaques connues
pour tester le systeme IDS en conditions reelles.

Usage:
    python3 demo_attacks.py --attack portscan    # Simule un port scan
    python3 demo_attacks.py --attack bruteforce  # Simule un brute force SSH
    python3 demo_attacks.py --attack dos         # Simule un DoS flooding
    python3 demo_attacks.py --attack benign       # Trafic normal de reference
    python3 demo_attacks.py --all                 # Lance toutes les attaques
    python3 demo_attacks.py --stream              # Mode streaming (generation continue)
"""
import os
import sys
import json
import argparse
import time
import random
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# ══════════════════════════════════════════════════════════════════════════════
# Generateurs d'attaques
# ══════════════════════════════════════════════════════════════════════════════

def generate_portscan(n_sessions=50, stealthy=False):
    """
    Simule une attaque de type Port Scan.

    Caracteristiques:
    - Nombreuses connexions depuis une meme source
    - Ports destination varies (1-65535)
    - Faible volume de donnees
    - Reponses absentes ou minimales (SYN scan)

    Args:
        n_sessions: Nombre de sessions a generer
        stealthy: Si True, utilise des scans lents (plus difficiles a detecter)

    Returns:
        list[dict]: Liste de sessions au format JSON
    """
    sessions = []
    scan_type = "Stealth" if stealthy else "Aggressive"
    print(f"[PortScan/{scan_type}] Generation de {n_sessions} sessions...")

    for i in range(n_sessions):
        if stealthy:
            # Stealth: scans lents, resembles normal traffic
            duration = np.random.lognormal(5, 1.5)
            fwd_packets = np.random.poisson(5)
            bwd_packets = np.random.poisson(3)
            fwd_len = int(np.clip(np.random.lognormal(8, 2), 0, 1e8))
            bwd_len = int(np.clip(np.random.lognormal(7, 2), 0, 1e8))
            syn_count = np.random.poisson(2)
            rst_count = 0
            ack_count = int(np.clip(np.random.lognormal(3, 1), 0, 1e5))
            dst_port = random.randint(1, 65535)
        else:
            # Aggressive: classic port scan
            duration = np.random.lognormal(2, 2)
            fwd_packets = np.random.poisson(1)
            bwd_packets = np.random.poisson(0)
            fwd_len = int(np.clip(np.random.lognormal(2, 1), 0, 1e8))
            bwd_len = 0
            syn_count = np.random.poisson(1)
            rst_count = np.random.poisson(0)
            ack_count = 0
            dst_port = random.randint(1, 65535)

        session = {
            "session_id": f"PS-{i+1:04d}",
            "attack_type": "PortScan",
            "scan_mode": scan_type,
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.50",
            "src_port": random.randint(40000, 60000),
            "dst_port": dst_port,
            "protocol": "TCP",
            "duration": round(float(duration), 2),
            "fwd_packets": int(fwd_packets),
            "bwd_packets": int(bwd_packets),
            "fwd_len": int(fwd_len),
            "bwd_len": int(bwd_len),
            "flow_bytes_s": round(float(np.clip(np.random.lognormal(2, 1.5), 1, 1e8)), 2),
            "flow_packets_s": round(float(np.clip(np.random.lognormal(1, 1), 0.1, 1e5)), 4),
            "fwd_iat_mean": round(float(np.clip(np.random.lognormal(1 if not stealthy else 4, 1 if not stealthy else 2), 0, 1e6)), 2),
            "bwd_iat_mean": round(float(np.clip(np.random.lognormal(0.1, 0.1), 0, 1e6)), 2),
            "fwd_psh_flags": 0,
            "syn_count": int(syn_count),
            "rst_count": int(rst_count),
            "ack_count": int(ack_count),
            "psh_count": 0,
            "active_mean": round(float(np.clip(np.random.lognormal(0.1 if not stealthy else 5, 0.1 if not stealthy else 2), 0, 1e6)), 2),
            "idle_mean": round(float(np.clip(np.random.lognormal(0.1 if not stealthy else 6, 0.1 if not stealthy else 2), 0, 1e6)), 2),
            "flow_iat_mean": round(float(np.clip(np.random.lognormal(1 if not stealthy else 4, 1 if not stealthy else 2), 0, 1e6)), 2),
            "fwd_len_mean": round(float(np.clip(np.random.lognormal(2 if not stealthy else 7, 1 if not stealthy else 2), 0, 1e6)), 2),
            "bwd_len_mean": round(float(bwd_len), 2),
            "_ground_truth": "attack"
        }
        sessions.append(session)

    return sessions


def generate_bruteforce(n_sessions=50, stealthy=False):
    """
    Simule une attaque de type Brute Force SSH/FTP.

    Caracteristiques:
    - Nombreuses connexions consecutives au port 22/21
    - Courte duree par connexion
    - Flags SYN+ACK repetes
    - echecs d'authentification (rst_count eleve)

    Args:
        n_sessions: Nombre de sessions a generer
        stealthy: Si True, utilise un rythme plus lent
    """
    sessions = []
    mode = "Stealth" if stealthy else "Aggressive"
    print(f"[BruteForce/{mode}] Generation de {n_sessions} sessions...")

    for i in range(n_sessions):
        if stealthy:
            duration = np.random.lognormal(4, 1)
            fwd_packets = np.random.poisson(8)
            bwd_packets = np.random.poisson(6)
            fwd_len = int(np.clip(np.random.lognormal(9, 1.5), 0, 1e8))
            bwd_len = int(np.clip(np.random.lognormal(10, 1.5), 0, 1e8))
            syn_count = np.random.poisson(2)
            rst_count = np.random.poisson(0)
            ack_count = int(np.clip(np.random.lognormal(4, 1), 0, 1e5))
            psh_count = np.random.poisson(1)
            active_mean = np.clip(np.random.lognormal(7, 1.5), 0, 1e6)
            idle_mean = np.clip(np.random.lognormal(8, 1.5), 0, 1e6)
            flow_iat = np.clip(np.random.lognormal(5, 1.5), 0, 1e6)
        else:
            duration = np.random.lognormal(1, 1.5)
            fwd_packets = np.random.poisson(3)
            bwd_packets = np.random.poisson(2)
            fwd_len = int(np.clip(np.random.lognormal(4, 2), 0, 1e8))
            bwd_len = int(np.clip(np.random.lognormal(4, 2), 0, 1e8))
            syn_count = np.random.poisson(8)
            rst_count = np.random.poisson(1)
            ack_count = int(np.clip(np.random.lognormal(2, 1), 0, 1e5))
            psh_count = np.random.poisson(0)
            active_mean = np.clip(np.random.lognormal(1, 1), 0, 1e6)
            idle_mean = np.clip(np.random.lognormal(0.5, 0.5), 0, 1e6)
            flow_iat = np.clip(np.random.lognormal(1.5, 1), 0, 1e6)

        session = {
            "session_id": f"BF-{i+1:04d}",
            "attack_type": "BruteForce",
            "brute_mode": mode,
            "src_ip": f"203.0.113.{random.randint(1,254)}",
            "dst_ip": "192.168.1.10",
            "src_port": random.randint(40000, 60000),
            "dst_port": random.choice([22, 21, 23]),
            "protocol": "TCP",
            "duration": round(float(duration), 2),
            "fwd_packets": int(fwd_packets),
            "bwd_packets": int(bwd_packets),
            "fwd_len": int(fwd_len),
            "bwd_len": int(bwd_len),
            "flow_bytes_s": round(float(np.clip(np.random.lognormal(12, 1.5), 1, 1e8)), 2),
            "flow_packets_s": round(float(np.clip(np.random.lognormal(3, 1.5), 0.1, 1e5)), 4),
            "fwd_iat_mean": round(float(np.clip(np.random.lognormal(5, 1.5), 0, 1e6)), 2),
            "bwd_iat_mean": round(float(np.clip(np.random.lognormal(6, 1.5), 0, 1e6)), 2),
            "fwd_psh_flags": int(psh_count),
            "syn_count": int(syn_count),
            "rst_count": int(rst_count),
            "ack_count": int(ack_count),
            "psh_count": int(psh_count),
            "active_mean": round(float(active_mean), 2),
            "idle_mean": round(float(idle_mean), 2),
            "flow_iat_mean": round(float(flow_iat), 2),
            "fwd_len_mean": round(float(np.clip(np.random.lognormal(8, 1.5), 0, 1e6)), 2),
            "bwd_len_mean": round(float(np.clip(np.random.lognormal(9, 1.5), 0, 1e6)), 2),
            "_ground_truth": "attack"
        }
        sessions.append(session)

    return sessions


def generate_dos(n_sessions=50, stealthy=False):
    """
    Simule une attaque de type DoS (SYN Flood, UDP Flood, etc.).

    Caracteristiques:
    - Tres haut debit de paquets
    - Courte duree (burst)
    - Ratio packets/data eleve
    - Destination souvent port 80/443/0

    Args:
        n_sessions: Nombre de sessions a generer
        stealthy: Si True, low-rate DoS plus difficile a detecter
    """
    sessions = []
    mode = "Low-Rate" if stealthy else "Flood"
    print(f"[DoS/{mode}] Generation de {n_sessions} sessions...")

    for i in range(n_sessions):
        if stealthy:
            # Low-rate DoS: simulates intermittent congestion
            duration = np.random.lognormal(4, 2)
            fwd_packets = np.random.poisson(15)
            bwd_packets = np.random.poisson(5)
            fwd_len = int(np.clip(np.random.lognormal(9, 2), 0, 1e8))
            bwd_len = int(np.clip(np.random.lognormal(8, 2), 0, 1e8))
            syn_count = np.random.poisson(5)
            rst_count = np.random.poisson(1)
            ack_count = int(np.clip(np.random.lognormal(4, 1), 0, 1e5))
            psh_count = np.random.poisson(2)
            active_mean = np.clip(np.random.lognormal(5, 2), 0, 1e6)
            idle_mean = np.clip(np.random.lognormal(6, 2), 0, 1e6)
            flow_iat = np.clip(np.random.lognormal(3.5, 2), 0, 1e6)
            dst_port = random.choice([80, 443, 53, 8080])
        else:
            duration = np.random.lognormal(1.5, 1.5)
            fwd_packets = np.random.poisson(100)
            bwd_packets = np.random.poisson(5)
            fwd_len = int(np.clip(np.random.lognormal(6, 3), 0, 1e8))
            bwd_len = int(np.clip(np.random.lognormal(2, 1), 0, 1e8))
            syn_count = np.random.poisson(50)
            rst_count = np.random.poisson(5)
            ack_count = int(np.clip(np.random.lognormal(5, 2), 0, 1e5))
            psh_count = np.random.poisson(10)
            active_mean = np.clip(np.random.lognormal(0.5, 0.5), 0, 1e6)
            idle_mean = np.clip(np.random.lognormal(0.1, 0.1), 0, 1e6)
            flow_iat = np.clip(np.random.lognormal(0.5, 0.5), 0, 1e6)
            dst_port = random.choice([80, 443, 53, 0])

        session = {
            "session_id": f"DS-{i+1:04d}",
            "attack_type": "DoS",
            "dos_mode": mode,
            "src_ip": f"{random.choice(['192.168','10.0','172.16'])}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}",
            "dst_ip": "10.0.0.1",
            "src_port": random.randint(1, 65535),
            "dst_port": dst_port,
            "protocol": "TCP",
            "duration": round(float(duration), 2),
            "fwd_packets": int(fwd_packets),
            "bwd_packets": int(bwd_packets),
            "fwd_len": int(fwd_len),
            "bwd_len": int(bwd_len),
            "flow_bytes_s": round(float(np.clip(np.random.lognormal(15, 3), 1, 1e8)), 2),
            "flow_packets_s": round(float(np.clip(np.random.lognormal(8, 2), 0.1, 1e5)), 4),
            "fwd_iat_mean": round(float(np.clip(np.random.lognormal(0.5 if not stealthy else 3, 0.5 if not stealthy else 2), 0, 1e6)), 2),
            "bwd_iat_mean": round(float(np.clip(np.random.lognormal(2 if not stealthy else 4, 1 if not stealthy else 2), 0, 1e6)), 2),
            "fwd_psh_flags": int(psh_count),
            "syn_count": int(syn_count),
            "rst_count": int(rst_count),
            "ack_count": int(ack_count),
            "psh_count": int(psh_count),
            "active_mean": round(float(active_mean), 2),
            "idle_mean": round(float(idle_mean), 2),
            "flow_iat_mean": round(float(flow_iat), 2),
            "fwd_len_mean": round(float(np.clip(np.random.lognormal(5 if not stealthy else 8, 2), 0, 1e6)), 2),
            "bwd_len_mean": round(float(np.clip(np.random.lognormal(1 if not stealthy else 7, 1 if not stealthy else 2), 0, 1e6)), 2),
            "_ground_truth": "attack"
        }
        sessions.append(session)

    return sessions


def generate_benign(n_sessions=50):
    """
    Genere du trafic reseau normal de reference.
    Sert de baseline pour comparer les detections.

    Returns:
        list[dict]: Liste de sessions normales
    """
    print(f"[Benign] Generation de {n_sessions} sessions normales...")
    sessions = []

    for i in range(n_sessions):
        duration = np.random.lognormal(4.5, 2)
        fwd_packets = np.random.poisson(8)
        bwd_packets = np.random.poisson(6)
        fwd_len = int(np.clip(np.random.lognormal(9, 2), 0, 1e8))
        bwd_len = int(np.clip(np.random.lognormal(10, 2), 0, 1e8))
        syn_count = np.random.poisson(2)
        rst_count = np.random.poisson(0)
        ack_count = int(np.clip(np.random.lognormal(3, 1), 0, 1e5))
        psh_count = np.random.poisson(1)
        active_mean = np.clip(np.random.lognormal(7, 2), 0, 1e6)
        idle_mean = np.clip(np.random.lognormal(8, 2), 0, 1e6)
        flow_iat = np.clip(np.random.lognormal(5.5, 2), 0, 1e6)

        session = {
            "session_id": f"BN-{i+1:04d}",
            "attack_type": "Benign",
            "src_ip": f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
            "dst_ip": f"10.0.{random.randint(1,254)}.{random.randint(1,254)}",
            "src_port": random.randint(1024, 65535),
            "dst_port": random.choice([80, 443, 22, 53, 8080]),
            "protocol": "TCP",
            "duration": round(float(duration), 2),
            "fwd_packets": int(fwd_packets),
            "bwd_packets": int(bwd_packets),
            "fwd_len": int(fwd_len),
            "bwd_len": int(bwd_len),
            "flow_bytes_s": round(float(np.clip(np.random.lognormal(12, 2), 1, 1e8)), 2),
            "flow_packets_s": round(float(np.clip(np.random.lognormal(3, 1.5), 0.1, 1e5)), 4),
            "fwd_iat_mean": round(float(np.clip(np.random.lognormal(5, 2), 0, 1e6)), 2),
            "bwd_iat_mean": round(float(np.clip(np.random.lognormal(6, 2), 0, 1e6)), 2),
            "fwd_psh_flags": int(psh_count),
            "syn_count": int(syn_count),
            "rst_count": int(rst_count),
            "ack_count": int(ack_count),
            "psh_count": int(psh_count),
            "active_mean": round(float(active_mean), 2),
            "idle_mean": round(float(idle_mean), 2),
            "flow_iat_mean": round(float(flow_iat), 2),
            "fwd_len_mean": round(float(np.clip(np.random.lognormal(8, 2), 0, 1e6)), 2),
            "bwd_len_mean": round(float(np.clip(np.random.lognormal(9, 2), 0, 1e6)), 2),
            "_ground_truth": "normal"
        }
        sessions.append(session)

    return sessions


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation avec le modele IDS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_sessions(sessions, dataset="cicids2017"):
    """
    Evalue une liste de sessions avec le modele IDS.
    Affiche un rapport de detection.

    Args:
        sessions: Liste de sessions generees
        dataset: Dataset a utiliser pour l'evaluation
    """
    import explain

    print(f"\n{'='*60}")
    print(f"Evaluation IDS (dataset: {dataset})")
    print(f"{'='*60}")

    tp = fp = tn = fn = 0

    for session in sessions:
        # Retirer la cle de ground truth avant inference
        session_for_inference = {k: v for k, v in session.items() if not k.startswith("_")}
        result = explain.explain_prediction(session_for_inference, dataset=dataset)

        pred = result["prediction"]
        gt = session.get("_ground_truth", "unknown")
        conf = result["confiance"]

        is_attack = pred == "ATTACK"
        is_gt_attack = gt == "attack"

        if is_attack and is_gt_attack:
            tp += 1
            status = "✅ TP"
        elif is_attack and not is_gt_attack:
            fp += 1
            status = "❌ FP"
        elif not is_attack and not is_gt_attack:
            tn += 1
            status = "✅ TN"
        else:
            fn += 1
            status = "❌ FN"

        print(f"  {session['session_id']} | {session['attack_type']:12s} | {pred:6s} "
              f"| conf={conf:.2%} | {status}")

    total = len(sessions)
    n_attacks = sum(1 for s in sessions if s.get("_ground_truth") == "attack")
    n_normal = total - n_attacks

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*60}")
    print(f"Resume  -  {sessions[0]['attack_type'] if sessions else '?'}")
    print(f"{'='*60}")
    print(f"  Total         : {total}")
    print(f"  Attaques reelles: {n_attacks}")
    print(f"  Trafic normal  : {n_normal}")
    print(f"  Vrais Positifs : {tp}")
    print(f"  Faux Positifs  : {fp}")
    print(f"  Vrais Negatifs : {tn}")
    print(f"  Faux Negatifs  : {fn}")
    print(f"  Accuracy       : {accuracy:.2%}")
    print(f"  Precision     : {precision:.2%}")
    print(f"  Recall        : {recall:.2%}")
    print(f"  F1-Score      : {f1:.2%}")
    print(f"{'='*60}\n")

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1}


# ══════════════════════════════════════════════════════════════════════════════
# Mode streaming
# ══════════════════════════════════════════════════════════════════════════════

def stream_attacks(dataset="cicids2017", interval=1.0, duration=60):
    """
    Mode streaming: generation continue de sessions attaquees
    pour tester le systeme IDS en conditions dynamiques.

    Args:
        dataset: Dataset a utiliser
        interval: Intervalle entre chaque session (secondes)
        duration: Duree totale du test (secondes)
    """
    import explain

    print(f"\n{'='*60}")
    print(f"MODE STREAMING  -  Dataset: {dataset}  -  Duree: {duration}s")
    print(f"{'='*60}")
    print("Appuyez sur Ctrl+C pour arreter.\n")

    attack_types = [
        ("Benign", lambda: generate_benign(1)),
        ("PortScan", lambda: generate_portscan(1, stealthy=False)),
        ("BruteForce", lambda: generate_bruteforce(1, stealthy=False)),
        ("DoS", lambda: generate_dos(1, stealthy=False)),
    ]

    start = time.time()
    session_num = 0

    try:
        while time.time() - start < duration:
            # Alterner les types d'attaques
            attack_name, gen_fn = random.choice(attack_types)
            sessions = gen_fn()
            session = sessions[0]

            session_for_inference = {k: v for k, v in session.items()
                                     if not k.startswith("_")}
            result = explain.explain_prediction(session_for_inference, dataset=dataset)

            session_num += 1
            elapsed = time.time() - start
            pred = result["prediction"]
            conf = result["confiance"]
            gt = session.get("_ground_truth", "?")

            is_correct = (pred == "ATTACK" and gt == "attack") or (pred == "NORMAL" and gt == "normal")
            status = "✅" if is_correct else "❌"

            print(f"[T+{elapsed:5.1f}s] #{session_num:04d} | {attack_name:12s} "
                  f"| {pred:6s} (conf={conf:.2%}) | GT={gt:6s} | {status}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStream arrete par l'utilisateur.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Demonstration d'attaques simulees pour le systeme IDS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python3 demo_attacks.py --attack portscan --dataset cicids2017
  python3 demo_attacks.py --attack bruteforce --stealth --dataset nslkdd
  python3 demo_attacks.py --all --dataset cicids2017
  python3 demo_attacks.py --stream --interval 0.5 --duration 30
        """
    )
    parser.add_argument("--attack", type=str,
                        choices=["portscan", "bruteforce", "dos", "benign"],
                        help="Type d'attaque a generer")
    parser.add_argument("--stealth", action="store_true",
                        help="Mode stealth (attaques plus difficiles a detecter)")
    parser.add_argument("--n", type=int, default=50,
                        help="Nombre de sessions par type (defaut: 50)")
    parser.add_argument("--all", action="store_true",
                        help="Lance toutes les attaques et le trafic normal")
    parser.add_argument("--stream", action="store_true",
                        help="Mode streaming (generation continue)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Intervalle entre sessions en mode stream (defaut: 1.0s)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duree du mode stream en secondes (defaut: 60)")
    parser.add_argument("--dataset", type=str, default="cicids2017",
                        choices=["nslkdd", "cicids2017"],
                        help="Dataset a utiliser (defaut: cicids2017)")
    parser.add_argument("--output", type=str,
                        help="Fichier JSON de sortie (optionnel)")

    args = parser.parse_args()

    if args.all:
        # Lancer toutes les attaques
        all_results = {}
        for attack_name in ["benign", "portscan", "bruteforce", "dos"]:
            gen_fn = {
                "benign": lambda: generate_benign(args.n),
                "portscan": lambda: generate_portscan(args.n, args.stealth),
                "bruteforce": lambda: generate_bruteforce(args.n, args.stealth),
                "dos": lambda: generate_dos(args.n, args.stealth),
            }[attack_name]
            sessions = gen_fn()
            result = evaluate_sessions(sessions, dataset=args.dataset)
            all_results[attack_name] = result

        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"Resultats sauvegardes dans {args.output}")

    elif args.stream:
        stream_attacks(dataset=args.dataset, interval=args.interval,
                       duration=args.duration)

    elif args.attack:
        gen_fn = {
            "benign": lambda: generate_benign(args.n),
            "portscan": lambda: generate_portscan(args.n, args.stealth),
            "bruteforce": lambda: generate_bruteforce(args.n, args.stealth),
            "dos": lambda: generate_dos(args.n, args.stealth),
        }[args.attack]
        sessions = gen_fn()

        if args.output:
            with open(args.output, "w") as f:
                json.dump(sessions, f, indent=2, default=str)
            print(f"Sessions sauvegardees dans {args.output}")
        else:
            result = evaluate_sessions(sessions, dataset=args.dataset)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
