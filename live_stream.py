#!/usr/bin/env python3
"""
Live Stream IDS - Module de capture et analyse en temps reel.

Fonctionnalites:
  1. Sniff direct avec Scapy (interface reseau)
  2. Lecture de fichier PCAP
  3. Ecoute socket TCP (localhost:9999) pour logs JSON (Zeek/Suricata)

Format JSON attendu par session:
{
  "session_id": "string",
  "src_ip": "string",
  "dst_ip": "string",
  "src_port": int,
  "dst_port": int,
  "protocol": "TCP"|"UDP"|"ICMP",
  "bytes_sent": int,
  "bytes_received": int,
  "packets": int,
  "duration": float,
  "flags": "string"
}

使用方法:
  # Mode socket (recommandé pour integration)
  python live_stream.py --mode socket --port 9999

  # Mode pcap
  python live_stream.py --mode pcap --file capture.pcap

  # Mode sniff (necessite droits root)
  python live_stream.py --mode sniff --interface eth0

Integration reseau reel:
  - Zeek: ajouter 'LogWriter::JSON' dans zeekctl et envoyer sur localhost:9999
  - Suricata: configurer EVE-JSON sur localhost:9999
  - Scapy: utiliser le script externe send_sessions.py
"""
import os
import sys
import json
import time
import socket
import argparse
import threading
import statistics
import pickle
from collections import deque

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# --- Palette cyber ---
COLOR_ATTACK = "#FF4B4B"
COLOR_NORMAL = "#00CC88"
COLOR_ACCENT = "#4B9FFF"
COLOR_BG = "#0E1117"

# ============================================================
# PARTIE 1: Conversion session reseau -> vecteur modele
# ============================================================

# Mapping protocol -> valeur encodee (doit correspondre au LabelEncoder)
KNOWN_PROTOCOLS = ["tcp", "udp", "icmp", "arp", "gre", "pptp"]
KNOWN_SERVICES = [
    "http", "https", "ftp", "ssh", "telnet", "smtp", "dns",
    "dhcp", "pop3", "nntp", "netbios_ns", "netbios_dgm", "netbios_ssn",
    "imap3", "ldap", "mysql", "oracle", "mssql", "postgres", "vnc", "x11",
    "snmp", "irc", "radius", "pop3s", "submission", "other"
]
KNOWN_FLAGS = [
    "SF", "S0", "REJ", "RSTO", "RSTR", "SH", "RSTOS0",
    "S1", "S2", "S3", "S4", "SYN", "ACK", "FIN", "OTH"
]

def session_to_features(session_dict):
    """
    Convertit un dictionnaire session (format live_stream) en vecteur
    de features compatible avec le modele IDS.

    Args:
        session_dict: dict avec les champs du format JSON attendu

    Returns:
        dict avec les 20 features attendues par le modele
    """
    protocol = session_dict.get("protocol", "TCP").lower()
    flags = session_dict.get("flags", "SF")
    bytes_sent = session_dict.get("bytes_sent", 0)
    bytes_received = session_dict.get("bytes_received", 0)
    duration = max(session_dict.get("duration", 1), 0.001)
    packets = max(session_dict.get("packets", 1), 1)
    src_port = session_dict.get("src_port", 0)
    dst_port = session_dict.get("dst_port", 0)

    # Heuristiques derivees (similaires NSL-KDD)
    dst_bytes = bytes_received
    src_bytes = bytes_sent

    # land: meme IP/PORT source et destination
    land = 1 if (session_dict.get("src_ip") == session_dict.get("dst_ip")
                 and src_port == dst_port) else 0

    # wrong_fragment
    wrong_fragment = 0

    # urgent
    urgent = 0

    # hot: score base sur ports suspects
    suspicious_ports = {20, 21, 23, 25, 110, 135, 139, 143, 445, 1433, 3306, 5432}
    hot = 1 if dst_port in suspicious_ports else 0

    # logged_in: guess based on auth ports
    auth_ports = {22, 23, 3389, 21, 110, 143}
    logged_in = 1 if dst_port in auth_ports else 0

    # num_compromised: heuristic based on bytes anomaly
    num_compromised = 1 if (src_bytes < 100 and dst_bytes > 10000) else 0

    # count: connexions a memes host (estimate)
    count = min(int(packets / 2), 255)

    # srv_count
    srv_count = min(int(packets / 4), 255)

    # serror_rate: Syn-only start pattern heuristic
    serror_rate = 1.0 if flags in ["S0", "REJ", "RSTR"] else 0.0

    # srv_serror_rate
    srv_serror_rate = serror_rate * 0.8

    # rerror_rate: reset received
    rerror_rate = 1.0 if flags in ["RSTO", "RSTR", "RSTOS0"] else 0.0

    # srv_rerror_rate
    srv_rerror_rate = rerror_rate * 0.8

    # diff_srv_rate: heuristic
    diff_srv_rate = 0.3 if protocol == "tcp" else 0.1

    # dst_host_count
    dst_host_count = min(count, 200)

    return {
        "duration": duration,
        "protocol_type": protocol.upper(),
        "service": _guess_service(dst_port),
        "flag": flags,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "land": land,
        "wrong_fragment": wrong_fragment,
        "urgent": urgent,
        "hot": hot,
        "logged_in": logged_in,
        "num_compromised": num_compromised,
        "count": count,
        "srv_count": srv_count,
        "serror_rate": serror_rate,
        "srv_serror_rate": srv_serror_rate,
        "rerror_rate": rerror_rate,
        "srv_rerror_rate": rerror_rate,
        "diff_srv_rate": diff_srv_rate,
        "dst_host_count": dst_host_count,
    }

def _guess_service(port):
    """Guess service name from port number."""
    port_services = {
        20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet",
        25: "smtp", 53: "domain", 80: "http", 110: "pop3",
        143: "imap3", 443: "https", 445: "microsoft_ds",
        3306: "mysql", 3389: "msrpc", 5432: "postgres"
    }
    return port_services.get(port, "other")

# ============================================================
# PARTIE 2: Source de sessions (socket, pcap, sniff)
# ============================================================

class SessionSource:
    """Interface abstraite pour les sources de sessions."""

    def __init__(self, alert_callback=None, threshold=0.7):
        self.alert_callback = alert_callback
        self.threshold = threshold
        self.running = False
        self.stats = {"total": 0, "attacks": 0, "alerts": 0}
        self.alert_history = deque(maxlen=1000)
        self.alerts_per_minute = deque(maxlen=60)  # 60 minutes de history

    def start(self):
        raise NotImplementedError

    def stop(self):
        self.running = False

    def get_stats(self):
        return dict(self.stats)

    def get_alert_history(self):
        return list(self.alert_history)

    def get_alerts_per_minute(self):
        return list(self.alerts_per_minute)


class SocketSource(SessionSource):
    """
    Ecoute les logs JSON sur une socket TCP.
    Accepte les connexions de Zeek, Suricata, ou tout outil envoyant du JSON.
    """

    def __init__(self, host="127.0.0.1", port=9999, **kwargs):
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.server_socket = None
        self._thread = None

    def start(self):
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)
        print(f"[SocketSource] Listening on {self.host}:{self.port}")
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self):
        while self.running:
            try:
                client, addr = self.server_socket.accept()
                threading.Thread(
                    target=self._handle_client,
                    args=(client, addr),
                    daemon=True
                ).start()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[SocketSource] Accept error: {e}")
                break

    def _handle_client(self, client, addr):
        try:
            buf = ""
            while self.running:
                data = client.recv(4096)
                if not data:
                    break
                buf += data.decode("utf-8", errors="ignore")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        self._process_line(line)
        except Exception as e:
            print(f"[SocketSource] Client error: {e}")
        finally:
            client.close()

    def _process_line(self, line):
        try:
            session = json.loads(line)
            self._analyze_session(session)
        except json.JSONDecodeError:
            pass

    def _analyze_session(self, session):
        from explain import explain_prediction
        self.stats["total"] += 1
        features = session_to_features(session)

        try:
            result = explain_prediction(features)
        except Exception:
            return

        pred = result["prediction"]
        score = result["confiance"]

        if pred == "attaque":
            self.stats["attacks"] += 1

        if score >= self.threshold:
            self.stats["alerts"] += 1
            alert = {
                "timestamp": time.time(),
                "session_id": session.get("session_id", "?"),
                "src_ip": session.get("src_ip", "?"),
                "dst_ip": session.get("dst_ip", "?"),
                "src_port": session.get("src_port", 0),
                "dst_port": session.get("dst_port", 0),
                "protocol": session.get("protocol", "?"),
                "score": score,
                "prediction": pred,
                "top_factors": result.get("top_facteurs", [])[:3],
            }
            self.alert_history.append(alert)

            # Update alerts/minute counter
            now = time.time()
            recent = sum(1 for a in self.alert_history if now - a["timestamp"] < 60)
            self.alerts_per_minute.append({"time": now, "count": recent})

            if self.alert_callback:
                self.alert_callback(alert)


class SimulatedSource(SessionSource):
    """
    Source simulee pour tests et demonstrations.
    Genere des sessions aleatoires avec un ratio configurable d'attaques.
    """

    def __init__(self, attack_ratio=0.30, interval=1.0, **kwargs):
        super().__init__(**kwargs)
        self.attack_ratio = attack_ratio
        self.interval = interval
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._generate_loop, daemon=True)
        self._thread.start()
        print(f"[SimulatedSource] Running (attack_ratio={self.attack_ratio}, interval={self.interval}s)")

    def _generate_loop(self):
        import random
        counter = 0
        while self.running:
            session = self._generate_session(counter)
            self._analyze_session(session)
            counter += 1
            time.sleep(self.interval)

    def _generate_session(self, idx):
        import random
        is_attack = random.random() < self.attack_ratio

        src_ip = f"192.168.1.{random.randint(2, 254)}"
        dst_ip = f"10.0.0.{random.randint(2, 254)}" if is_attack else f"8.8.8.{random.randint(2, 254)}"

        if is_attack:
            return {
                "session_id": f"sim-{idx:06d}",
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": random.randint(49152, 65535),
                "dst_port": random.choice([21, 23, 25, 80, 139, 445, 1433]),
                "protocol": random.choice(["TCP", "TCP", "UDP"]),
                "bytes_sent": random.randint(0, 500),
                "bytes_received": random.randint(0, 500),
                "packets": random.randint(1, 10),
                "duration": random.uniform(0, 2),
                "flags": random.choice(["S0", "REJ", "RSTR", "SH", "RSTO"]),
            }
        else:
            return {
                "session_id": f"sim-{idx:06d}",
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": random.randint(49152, 65535),
                "dst_port": random.choice([80, 443, 22, 53, 8080]),
                "protocol": random.choice(["TCP", "UDP"]),
                "bytes_sent": random.randint(1000, 2000000),
                "bytes_received": random.randint(2000, 3000000),
                "packets": random.randint(5, 100),
                "duration": random.uniform(1, 60),
                "flags": random.choice(["SF", "S0", "SYN", "ACK"]),
            }

    def _analyze_session(self, session):
        from explain import explain_prediction
        self.stats["total"] += 1
        features = session_to_features(session)

        try:
            result = explain_prediction(features)
        except Exception:
            return

        pred = result["prediction"]
        score = result["confiance"]

        if pred == "attaque":
            self.stats["attacks"] += 1

        if score >= self.threshold:
            self.stats["alerts"] += 1
            alert = {
                "timestamp": time.time(),
                "session_id": session.get("session_id", "?"),
                "src_ip": session.get("src_ip", "?"),
                "dst_ip": session.get("dst_ip", "?"),
                "src_port": session.get("src_port", 0),
                "dst_port": session.get("dst_port", 0),
                "protocol": session.get("protocol", "?"),
                "score": score,
                "prediction": pred,
                "top_factors": result.get("top_facteurs", [])[:3],
            }
            self.alert_history.append(alert)

            now = time.time()
            recent = sum(1 for a in self.alert_history if now - a["timestamp"] < 60)
            self.alerts_per_minute.append({"time": now, "count": recent})

            if self.alert_callback:
                self.alert_callback(alert)


class PcapSource(SessionSource):
    """
    Lit un fichier PCAP et extrait les sessions pour analyse.
    Necessite Scapy: pip install scapy
    """

    def __init__(self, pcap_path, **kwargs):
        super().__init__(**kwargs)
        self.pcap_path = pcap_path

    def start(self):
        self.running = True
        threading.Thread(target=self._process_pcap, daemon=True).start()
        print(f"[PcapSource] Processing {self.pcap_path}")

    def _process_pcap(self):
        try:
            from scapy.all import rdpcap, TCP, UDP, ICMP, IP
        except ImportError:
            print("[PcapSource] Scapy non disponible. Installez-le: pip install scapy")
            return

        try:
            packets = rdpcap(self.pcap_path)
        except Exception as e:
            print(f"[PcapSource] Erreur lecture pcap: {e}")
            return

        sessions = {}
        for pkt in packets:
            if IP not in pkt:
                continue
            ip_src = pkt[IP].src
            ip_dst = pkt[IP].dst
            key = tuple(sorted([ip_src, ip_dst]))

            proto = "OTHER"
            src_port = 0
            dst_port = 0
            flags = ""
            bytes_sent = 0
            bytes_received = 0

            if TCP in pkt:
                proto = "TCP"
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
                flags = str(pkt[TCP].flags) if hasattr(pkt[TCP], "flags") else ""
                bytes_sent = len(pkt[TCP].payload) if hasattr(pkt[TCP], "payload") else 0
            elif UDP in pkt:
                proto = "UDP"
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
                bytes_sent = len(pkt[UDP].payload) if hasattr(pkt[UDP], "payload") else 0
            elif ICMP in pkt:
                proto = "ICMP"
                bytes_sent = len(pkt[ICMP].payload) if hasattr(pkt[ICMP], "payload") else 0

            if key not in sessions:
                sessions[key] = {
                    "session_id": f"pcap-{key[0]}-{key[1]}",
                    "src_ip": ip_src,
                    "dst_ip": ip_dst,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "protocol": proto,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                    "packets": 0,
                    "duration": 0,
                    "flags": flags,
                }

            sessions[key]["bytes_sent"] += bytes_sent
            sessions[key]["packets"] += 1
            sessions[key]["flags"] = flags

        for session in sessions.values():
            self._analyze_session(session)
            time.sleep(0.01)  # Avoid flooding

        print(f"[PcapSource] Finished: {len(sessions)} sessions extracted")


# ============================================================
# PARTIE 3: CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Live Stream IDS - Capture et analyse")
    parser.add_argument("--mode", choices=["socket", "simulate", "pcap"], default="simulate",
                        help="Mode de capture (socket=socket TCP, simulate=simulation, pcap=fichier pcap)")
    parser.add_argument("--host", default="127.0.0.1", help="Host pour mode socket")
    parser.add_argument("--port", type=int, default=9999, help="Port pour mode socket")
    parser.add_argument("--file", help="Chemin du fichier pcap (mode pcap)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Seuil d'alerte")
    parser.add_argument("--attack-ratio", type=float, default=0.3, help="Ratio d'attaques (simulation)")
    parser.add_argument("--interval", type=float, default=1.0, help="Intervalle secondes (simulation)")
    args = parser.parse_args()

    def print_alert(alert):
        print(f"[ALERT] {alert['timestamp']:.0f} | {alert['src_ip']}:{alert['src_port']} -> "
              f"{alert['dst_ip']}:{alert['dst_port']} | {alert['protocol']} | "
              f"Score: {alert['score']:.2%} | {alert['prediction'].upper()}")

    if args.mode == "socket":
        source = SocketSource(
            host=args.host, port=args.port,
            alert_callback=print_alert, threshold=args.threshold
        )
    elif args.mode == "pcap":
        source = PcapSource(args.file, alert_callback=print_alert, threshold=args.threshold)
    else:
        source = SimulatedSource(
            attack_ratio=args.attack_ratio, interval=args.interval,
            alert_callback=print_alert, threshold=args.threshold
        )

    source.start()
    print(f"[*] Live Stream IDS started in {args.mode} mode. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(5)
            stats = source.get_stats()
            print(f"[STATS] Total: {stats['total']} | Attacks: {stats['attacks']} | Alerts: {stats['alerts']}")
    except KeyboardInterrupt:
        print("\n[*] Stopping...")
        source.stop()


if __name__ == "__main__":
    main()
