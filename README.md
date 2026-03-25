# Cyber IDS Streamlit

Systeme de Detection d'Intrusion (IDS) base sur Machine Learning avec dashboard interactif.

## Description

Application web permettant de detecter des intrusions reseau en temps reel grace a des modeles Random Forest et XGBoost entranes sur le dataset NSL-KDD. Elle propose un dashboard 4 pages avec comparaison de modeles, simulation temps reel, test manuel et capture reseau live.

## Dataset

**NSL-KDD** (National Library of Knowledge Discovery)
- Source : https://www.unb.ca/cic/datasets/nslkdd.html
- 125 973 sessions d'entrainement, 22 544 de test
- 22 types d'attaques (Neptune, Smurf, Pod, Teardrop, etc.)
- Cible : `binary_label` (0 = normal, 1 = attaque)
- Features : duration, protocol_type, service, src_bytes, dst_bytes, serror_rate, etc.

## Stack technique

- **Python 3** + venv
- **scikit-learn** — Random Forest, preprocessing
- **XGBoost** — second modele
- **SHAP** — explicabilite locale et globale
- **Streamlit** — dashboard web
- **Plotly** — graphiques interactifs
- **Scapy / socket TCP** — capture live (optionnelle)

## Installation

```bash
cd ~/cyber-ids
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancement

```bash
source .venv/bin/activate
streamlit run app.py --server.headless true
```

Ouvrir : http://localhost:8501

## Les 4 pages

### 1. Modeles
Comparaison des performances RF vs XGBoost : accuracy, F1, AUC-ROC, matrices de confusion, importance des variables (SHAP/XGBoost).

### 2. Simulation RT
Generation de sessions reseau synthetiques avec distribution realist (basee sur NSL-KDD). Affichage temps reel ligne par ligne avec badge rouge/vert et score de confiance.

### 3. Tester une session
Formulaire de saisie complet (IP, ports, protocole, bytes, taux d'erreur, etc.) avec prediction + confiance + top 5 facteurs explicatifs en francais.

### 4. Live Stream
Capture temps reel sur 4 modes :
- **Simulation** : generation integree
- **Socket TCP** : ecoute sur localhost:9999
- **PCAP** : lecture de fichier capture
- **Sniff** : capture Scapy sur interface reseau

---

## Connexion a un reseau reel (pour le memoire)

Le systeme peut etre connecte a un reseau reel pour analyser du trafic en production. Trois integrations sont possibles :

### Option 1 — Scapy (sniff direct)

Capture directe des paquets sur une interface reseau. Necessite les droits root.

```python
# Script: send_sessions.py (exemple)
from scapy.all import sniff, IP, TCP, UDP
import socket, json

SOCKET_HOST = "localhost"
SOCKET_PORT = 9999

def packet_to_session(pkt):
    if IP in pkt:
        proto = "TCP" if TCP in pkt else ("UDP" if UDP in pkt else "ICMP")
        sport = pkt[TCP].sport if TCP in pkt else 0
        dport = pkt[TCP].dport if TCP in pkt else 0
        return {
            "session_id": f"SCAPY-{pkt.time}",
            "src_ip": pkt[IP].src,
            "dst_ip": pkt[IP].dst,
            "src_port": sport,
            "dst_port": dport,
            "protocol": proto,
            "bytes_sent": len(pkt),
            "bytes_received": 0,
            "packets": 1,
            "duration": 0.0,
            "flags": str(pkt[TCP].flags) if TCP in pkt else ""
        }
    return None

def send_to_app(session):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((SOCKET_HOST, SOCKET_PORT))
    s.send((json.dumps(session) + "\n").encode())
    s.close()

sniff(iface="eth0", prn=lambda p: [send_to_app(p) for s in [packet_to_session(p)] if s], store=0)
```

Lancer le dashboard en mode socket :
```bash
source .venv/bin/activate
streamlit run app.py --server.headless true
# selectionner "4. Live Stream" puis mode "Socket" et port 9999
```

### Option 2 — Zeek (IDS academique)

Zeek genere des logs de connexion stylises. Configurer l'emission JSON :

```
# /etc/zeek/local.zeek
@load protocols/http/detailed
Log::default_writer = Log::WRITER_ASCII([]);
```

Pour envoyer au dashboard, un script convertit les logs Zeek en JSON sur le socket :

```python
# zeek_to_socket.py
import socket, json, time
from pathlib import Path

def convert_zeek_log(line):
    fields = line.strip().split('\t')
    # positions selon conn.log de Zeek
    return {
        "session_id": f"ZEEK-{fields[0]}",
        "src_ip": fields[2], "dst_ip": fields[4],
        "src_port": int(fields[5]), "dst_port": int(fields[6]),
        "protocol": fields[7],
        "bytes_sent": int(fields[9]), "bytes_received": int(fields[10]),
        "duration": float(fields[11]),
        "flags": fields[8], "packets": 0
    }

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("localhost", 9999))
for line in Path("/var/log/zeek/conn.log").open():
    if line.startswith("#"): continue
    session = convert_zeek_log(line)
    s.send((json.dumps(session) + "\n").encode())
s.close()
```

### Option 3 — Suricata (EVE-JSON)

Suricata produit des logs JSON нормализés (EVE format). Configurer :

```yaml
# /etc/suricata/suricata.yaml
outputs:
  -eve-log:
      enabled: yes
      types:
        - alert
        - flow
      filename: eve.json
      types:
        - alert:
            payload: yes
            payload-printable: yes
      destination: socket
      socket:
        enabled: yes
        filename: /tmp/suricata.sock
```

Puis un parser envoit au dashboard :

```python
# suricata_to_socket.py
import socket, json, tailer

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("localhost", 9999))

for line in tailer.follow(open("/var/log/suricata/eve.json")):
    try:
        eve = json.loads(line)
        if eve.get("event_type") == "flow":
            f = eve["flow"]
            session = {
                "session_id": f"SURICATA-{eve.get('timestamp', '')}",
                "src_ip": f.get("src_ip", ""),
                "dst_ip": f.get("dest_ip", ""),
                "src_port": f.get("src_port", 0),
                "dst_port": f.get("dest_port", 0),
                "protocol": eve.get("proto", "TCP"),
                "bytes_sent": f.get("bytes_toclient", 0),
                "bytes_received": f.get("bytes_toserver", 0),
                "packets": f.get("pkts_toclient", 0) + f.get("pkts_toserver", 0),
                "duration": 0.0,
                "flags": ""
            }
            s.send((json.dumps(session) + "\n").encode())
    except: pass
```

### Format JSON attendu par le dashboard

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

Champs requis : session_id, src_ip, dst_ip, protocol.
Champs recommandes : src_port, dst_port, bytes_sent, bytes_received, duration, flags.

### Schema d'integration complete

```
[Reseau reel]
    |
    +-- Scapy (sniff) --> send_sessions.py --> [Socket :9999] --> Streamlit Live Stream
    +-- Zeek (conn.log) --> zeek_to_socket.py ---------> [Socket :9999] --> Streamlit Live Stream
    +-- Suricata (EVE-JSON) --> suricata_to_socket.py -> [Socket :9999] --> Streamlit Live Stream
    +-- Wireshark (pcap) --> tcpdump -w - > capture.pcap --> mode PCAP dans Streamlit
```

## Metriques des modeles

| Modele | Accuracy | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| Random Forest | 97.90% | 97.81% | 99.70% |
| XGBoost | 98.45% | 98.39% | 99.86% |

Meilleur modele : **XGBoost**

## Fichiers principaux

| Fichier | Role |
|---------|------|
| `app.py` | Dashboard Streamlit (4 pages) |
| `prep_data_nslkdd.py` | Telechargement et preprocessing NSL-KDD |
| `train.py` | Entrainement RF + XGBoost |
| `explain.py` | Explicabilite SHAP + fallback |
| `live_stream.py` | Module de capture reseau |
| `requirements.txt` | Dependances Python |
| `data/` | Dataset NSL-KDD (CSV) |
| `models/` | Modeles entranes (joblib) |
| `artifacts/` | Artefacts intermediaires |
| `logs/` | Journal d'execution |

## Limitations et ameliorations

**Limites actuelles :**
- Dataset NSL-KDD date de 2009 (reseau academique, moins representative du trafic actuel)
- Simulation temps reel basee sur des distributions statistiques, pas du vrai trafic
- Seuil d'alerte fixe (non adaptatif)

**Ameliorations possibles :**
- CICIDS2017 ou CSE-CIC-IDS2018 pour un dataset plus recent
- Integration Scapy/Zeek en production pour le mode live
- Retraining automatique sur nouveau trafic labelise
- Alerting (webhook, email, Slack) lors d'attaques
- Interface multi-utilisateurs avec historique
- Export PDF du rapport d'analyse

## Licence

MIT
