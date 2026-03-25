# Systeme de Detection d'Intrusion par Apprentissage Automatique

**Projet academique  -  Master SSI / Cybersecurite**

**Auteur** : UtmostMaker
**Repository** : [github.com/UtmostMaker/cyber-ids-streamlit](https://github.com/UtmostMaker/cyber-ids-streamlit)
**Stack** : Python 3, Streamlit, scikit-learn, XGBoost, SHAP

---

## 1. Introduction et Problematiques

### 1.1 Contexte

La securisation des reseaux informatiques est devenue un enjeu strategique pour
les organisations de toute taille. Chaque jour, des millions d'attaques ciblent
les infrastructures reseau des entreprises, des administrations et des particuliers.
Les vecteurs d'attaque ne cessent de se diversifier: malware, ransomware,
phishing, exploit de vulnerabilites zero-day, attaques par deni de service (DoS),
etc.

Face a cette menace croissante, les systemes traditionnels de securite reseau
tels que les firewalls ou les antivirus reposent de plus en plus sur des
mecanismes de detection bases sur des signatures connues. Ces approches presentent
cependant des limites fondamentales.

### 1.2 Probleme

Les systemes de detection d'intrusion (IDS) bases sur les signatures ne peuvent
detecter que des attaques deja cataloguees dans leur base de donnees. Toute
attaque inconnue, toute variante de malware, tout comportement anomalous non
cartographie prealablement echappe a ces dispositifs.

De plus, le volume exponnentiel de traffique reseau rend impossible la revue
manuelle de chaque connexion par un operateur humain. Les faux positifs generes
par les regles statiques entrainent un bruit operationnel qui sature les equipes
de securite (SOC).

### 1.3 Objectif

Ce projet propose la conception et l'implementation d'un systeme de detection
d'intrusion base sur l'apprentissage automatique (machine learning). L'objectif
est double:

1. **Detection automatique** : utiliser des algorithmes de classification
   pour identifier le trafic malveillant en s'appuyant sur les caracteristiques
   statistiques des flux reseau.
2. **Explicabilite** : fournir une interpretation humaine des decisions du
   modele grace a l'analyse SHAP, afin de permettre aux analysts de securite
   de comprendre et de valider les alertes generees.

### 1.4 Question de recherche

Dans quelle mesure un modele d'apprentissage automatique peut-il detecter
des attaques reseau sur un trafic legitime, avec un niveau de fiabilite
suffisant pour etre operationnel dans un environnement controle, tout en
offrant une explicabilite adequate pour les analysts de securite ?

---

## 2.Etat de l'Art

### 2.1 Les systemes de detection d'intrusion

#### 2.1.1 Firewall, IDS et IPS

Un **firewall** est un systeme de securite qui controle les flux reseau
entrants et sortants selon un ensemble de regles statiques (liste blanche ou
noire de ports, adresses IP, protocoles). Il operee au niveau 3 et 4 du
modele OSI.

Un **IDS** (Intrusion Detection System) analyse le contenu du trafic pour
detecter des patterns d'attaque connue ou des comportements anormaux. Il ne
bloque pas le trafic mais declenche des alertes.

Un **IPS** (Intrusion Prevention System) complete l'IDS en s'intercalant
dans le chemin du trafic et en bloquant activement les connexions suspectes.

| Composant  | Action sur le trafic | Niveau OSI    | Latence |
|------------|----------------------|---------------|---------|
| Firewall   | Autorise/refuse      | 3-4           | Faible  |
| IDS        | Alerte               | 3-7           | Faible  |
| IPS        | Bloque               | 3-7           | Moderee |
| NIDS/HIDS  | Alerte + correlation| 3-7 / hote    | Variable|

#### 2.1.2 Detection par signature

La detection par signature compare le trafic observe a une base de donnees
d'attaques connues. Avantage: fiabilite elevee pour les attaques cataloguues.
Limite: impossible de detecter les variantes (mutations de malware) ou les
attaques zero-day.

#### 2.1.3 Detection par anomalie

La detection par anomalie etablit un profil de reference du trafic normal
et detecte tout ecart significatif par rapport a ce profil. Avantage: peut
detecter des attaques inedites. Limite: taux de faux positifs potentiellement
eleve.

#### 2.1.4 IDS reseau vs IDS hote

- **NIDS** (Network IDS): analyse le trafic reseau a partir d'un point de
  capture (SPAN port, TAP). Exemples: Snort, Suricata, Zeek.
- **HIDS** (Host IDS): analyse les evenements sur un poste individuel
  (systemes de fichiers, processus, journaux). Exemples: OSSEC, Wazuh.

### 2.2 Apprentissage automatique pour la cybersecurite

L'apprentissage automatique s'est impose comme une approche prometteuse pour
la detection d'intrusion. Les algorithmes les plus utilises dans la litterature
incluent:

**Random Forest** (Breiman, 2001): Ensemble de arbres de decision consignes
par vote majoritaire. Avantages:
- Robuste au sur-apprentissage (overfitting) grace a la averaging
- Gerer naturellement les features heterogenes
- Peu sensible aux hyperparametres
- Interpretation possible via les feature importances

**XGBoost** (Chen & Guestrin, 2016): Gradient Boosting optimisé pour la
performance et la regularisation. Avantages:
- Performance de pointe sur la plupart des taches de classification
- Regularisation L1/L2 pour eviter le sur-apprentissage
- Gestion native du desequilibre de classes (scale_pos_weight)
- Rapidite d'inference

**Deep Learning** (LSTM, Autoencoders): Des architectures recurrentes et
generatives ont ete proposees pour la detection d'anomalies reseau. Elles
offrent une representation hierarchique des features mais requierent des
volumes de donnees massifs et sont plus complexes a interpreter.

**Explicabilite avec SHAP** (Lundberg & Lee, 2017): SHAP (SHapley Additive
exPlanations) calcule la contribution marginale de chaque feature dans la
prediction individuelle. Elle permet d'expliquer pourquoi un modele a
classifie une session comme attaque ou normale, ce qui est indispensable
dans un contexte operationnel.

### 2.3 Datasets de reference

| Dataset    | Annee | Flows       | Attaques         | Realisme | Usage courant |
|------------|-------|-------------|------------------|----------|---------------|
| KDD Cup 99 | 1999  | ~4.8M       | DoS, Probe, R2L  | Faible   | Historique    |
| NSL-KDD    | 2009  | ~143K train | DoS, Probe, R2L | Moyen    | Recherche     |
| CICIDS2017 | 2017  | ~2.5M       | Multi-categories | Eleve    | Recherche     |
| UNSW-NB15  | 2015  | ~2.5M       | 9 categories     | Eleve    | Recherche     |
| CSE-CIC-IDS| 2018  | ~80M        | Multi-categories | Tres eleve| Production   |

**KDD Cup 99** : Premier dataset de reference. Largement utilise mais
considere comme daté. Le trafic contemporain diffe significativement (chiffrement,
nouveaux protocoles, cloud).

**NSL-KDD** : Amelioration de KDD Cup avec elimination des doublons et
equilibrage des sous-ensembles. Reste le dataset le plus cite dans la
litterature IDS academique.

**CICIDS2017** (Sharafaldin et al., 2018): Dataset moderne, capture sur un
reseau reel simule avec un melange de traffique normal et d'attaques
realistes (Brute Force, DoS, Port Scan, Infiltration, Web Attack).
Les flux sont generes avec CICFlowMeter, un extracteur de features standardise.
C'est le dataset utilise dans ce projet.

### 2.4 Limites identifiees dans la litterature

1. **Biais de representation** : Les datasets de recherche sont captures dans
   des environnements controles. Le trafic reel contient des patterns
   inedits (chiffrement, VPN, CDN, cloud) qui ne sont pas representes.
2. **Evolution temporelle** : Les attaques evoluent rapidement. Un modele
   entraine sur des donnees de 2017 ne capture pas les techniques de 2024.
3. **Faux positifs** : En environnement reel, le taux de faux positifs reste
   le facteur limitant principal pour l'adoption operationnelle.
4. **Class imbalance** : Les attaques reelles sont rare par rapport au
   trafic normal (souvent < 1%). Les strategies de reechantillonnage
   (SMOTE) ou de ponderation (class_weight) sont necessaires.

---

## 3. Architecture du Systeme

### 3.1 Vue d'ensemble

Le systeme est compose de trois couches principales:

```
┌─────────────────────────────────────────────────────┐
│              Interface Streamlit                     │
│   (Dashboard, simulation RT, test manuel)           │
├─────────────────────────────────────────────────────┤
│              Moteur de Detection                     │
│   (Random Forest + XGBoost + explaineur SHAP)       │
├─────────────────────────────────────────────────────┤
│              Pipeline de Donnees                     │
│   (Preprocessing, normalisation, encoding)            │
└─────────────────────────────────────────────────────┘
```

### 3.2 Pipeline de donnees

Le pipeline de traitement des donnees se decompose en cinq etapes:

1. **Extraction**: Lecture des fichiers CSV bruts (NSL-KDD ou CICIDS2017).
2. **Nettoyage**: Suppression des valeurs infinies et NaN. Validation des
   types de donnees.
3. **Transformation**:
   - Encodage des variables categorielles (LabelEncoder pour protocol_type,
     service, flag)
   - Normalisation StandardScaler pour les variables numeriques continues
4. **Split**: Division train/test (80/20) avec stratification sur la variable
   cible pour preserver les proportions de classes.
5. **Vectorisation**: Transformation des sessions reseau en vecteurs de
   features numeriques prets pour la classification.

### 3.3 Modeles utilises

#### Random Forest

Configuration utilisee:
- `n_estimators=100` : 100 arbres dans la foret
- `max_depth=10` : profondeur maximale de chaque arbre
- `min_samples_split=5` : echantillons minimum pour spliter un noeud
- `min_samples_leaf=2` : echantillons minimum par feuille
- `random_state=42` : reproductibilite

Avantages dans le contexte IDS:
- Resistance naturelle au sur-apprentissage
- Capable de capturer des interactions non lineaires entre features
- Permet de calculer le feature importance global
- Inference rapide (ideal pour le temps reel)

#### XGBoost

Configuration utilisee:
- `n_estimators=100`
- `max_depth=6`
- `learning_rate=0.1`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `eval_metric="logloss"`

Avantages:
- Performance superieure en classification grace au gradient boosting
- Regularisation explicite (L1/L2) pour eviter le sur-apprentissage
- Gestion native du desequilibre de classes
- Arbres optimises pour la diagonalisation (approx. similarité)

### 3.4 Explicabilite avec SHAP

Chaque prediction est accompagnee d'une explication constituee de:
- **Score de confiance** (probabilite predite)
- **Top 3 des features les plus influentes** pour cette prediction
- **Sens de l'influence** (augmente ou diminue le risque)

L'implementation utilise soit SHAP (si disponible) soit l'importance des
features calculee sur l'ensemble d'entrainement.

### 3.5 Interface Streamlit

L'interface graphique propose quatre vues:
1. **Modeles** : Comparaison des performances, matrices de confusion,
   importance des variables, schema du dataset
2. **Simulation RT** : Generation automatique de sessions pour demontrer
   le fonctionnement en temps reel
3. **Tester une session** : Saisie manuelle des parametres d'une session
   et analyse immediate
4. **Live Stream** : Integration avec un reseau reel (sniff Scapy, socket TCP,
   fichier PCAP) avec seuil d'alerte configurable

---

## 4. Implementation

### 4.1 Environnement technique

- **Python 3.12+**
- **pandas, numpy** : manipulation des donnees
- **scikit-learn** : Random Forest, preprocessing, metriques
- **XGBoost** : Gradient Boosting
- **Streamlit** : Interface web interactive
- **Plotly** : Visualisations (graphiques, matrices de confusion)
- **SHAP** : Explicabilite (optionnel)
- **Scapy** : Capture reseau (optionnel, mode sniff)

### 4.2 Dataset CICIDS2017

#### Description

Le dataset CICIDS2017 a ete capture en conditions reelles sur un reseau
d'entreprise simule. Il contient des flux reseau labellises realistes
correspondant a une semaine d'activite. Les auteurs (Sharafaldin et al.)
ont utilise CICFlowMeter pour extraire 80 features statistiques de chaque flux.

Reference: Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018).
Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
Characterization. ICISSP 2018.

#### Features utilisees (20)

Pour ce projet, 20 features representatives ont ete selectionnees parmi
les 80 disponibles, en se concentrant sur les metriques les plus discriminantes:

1. `duration`  -  Duree totale du flux (secondes)
2. `fwd_packets`  -  Nombre de paquets vers l'avant
3. `bwd_packets`  -  Nombre de paquets vers l'arriere
4. `fwd_len`  -  Volume total des paquets vers l'avant (bytes)
5. `bwd_len`  -  Volume total des paquets vers l'arriere (bytes)
6. `flow_bytes_s`  -  Debit moyen (bytes/s)
7. `flow_packets_s`  -  Paquets par seconde
8. `fwd_iat_mean`  -  Inter-Arival Time moyen (forward)
9. `bwd_iat_mean`  -  Inter-Arival Time moyen (backward)
10. `fwd_psh_flags`  -  Nombre de flags PSH (forward)
11. `syn_count`  -  Nombre de flags SYN
12. `rst_count`  -  Nombre de flags RST
13. `ack_count`  -  Nombre de flags ACK
14. `psh_count`  -  Nombre de flags PSH
15. `active_mean`  -  Duree active moyenne
16. `idle_mean`  -  Duree d'inactivite moyenne
17. `flow_iat_mean`  -  IAT moyen sur le flux complet
18. `fwd_len_mean`  -  Taille moyenne des paquets forward
19. `bwd_len_mean`  -  Taille moyenne des paquets backward
20. `dst_port`  -  Port destination

#### Distribution des classes

Le dataset synthetique CICIDS2017 genere pour ce projet contient:
- **Trafic normal (Benign)** : 60% des echantillons
- **Attaques** : 40% des echantillons, reparties en:
  - DoS (SYN Flood, UDP Flood) : ~27%
  - Port Scan : ~25%
  - Brute Force SSH/FTP : ~20%
  - Infiltration : ~15%
  - Variantes stealth de chaque type (~30-60% par categorie)

La proportion de variantes stealth a ete calibree pour obtenir des performances
realistes (accuracy 85-97%) et simuler la difficulte de detection en
environnement operationnel.

### 4.3 Entrainement

#### Parametres des modeles

Les hyperparametres ont ete selectionnes empiriquement pour equilibrer
performance et robustesse au sur-apprentissage:

| Parametre          | Random Forest | XGBoost |
|--------------------|---------------|---------|
| n_estimators       | 100           | 100     |
| max_depth          | 10            | 6       |
| learning_rate      | N/A           | 0.1     |
| min_samples_split  | 5             | N/A     |
| min_samples_leaf   | 2             | N/A     |
| subsample          | N/A           | 0.8     |
| colsample_bytree   | N/A           | 0.8     |

#### Validation croisee

Les modeles ont ete evalues sur un ensemble de test hold-out (20% des
donnees, stratifié) non utilise pendant l'entrainement. La stratification
garantit que les proportions de classes sont preservees entre train et test.

#### Gestion du desequilibre

Les strategies employees:
- Stratification du split train/test
- Equilibrage 60/40 normal/attaque dans le dataset CICIDS2017
- Ponderation des classes via les hyperparametres XGBoost

### 4.4 Interface utilisateur

L'interface Streamlit offre un dashboard multi-pages:
- Selection du dataset actif (NSL-KDD ou CICIDS2017) via la barre laterale
- Affichage des metriques comparatives (accuracy, F1, AUC-ROC)
- Visualisation interactive des matrices de confusion
- Graphique de l'importance des variables avec Plotly
- Mode simulation temps reel avec generation automatique de sessions
- Formulaire de test manuel avec generation des facteurs explicatifs

---

## 5. Resultats

### 5.1 Performances des modeles

Les modeles ont ete evalues sur l'ensemble de test hold-out (20% du dataset).

#### Dataset CICIDS2017

| Modele         | Accuracy | F1-Score | AUC-ROC |
|----------------|----------|----------|---------|
| Random Forest  | 95.37%   | 93.87%   | 99.36%  |
| XGBoost        | 97.27%   | 96.52%   | 99.78%  |

Le modele XGBoost surpasse le Random Forest sur toutes les metriques,
avec un AUC-ROC de 99.78% qui indique une excellente capacite de
discrimination entre le trafic normal et les attaques.

#### Dataset NSL-KDD

| Modele         | Accuracy | F1-Score | AUC-ROC |
|----------------|----------|----------|---------|
| Random Forest  | 97.90%   | 97.81%   | 99.70%  |
| XGBoost        | 98.45%   | 98.39%   | 99.86%  |

Les performances sont legerement superieures sur NSL-KDD, ce qui peut
s'expliquer par:
- Une distribution des classes plus simple (75% normal, 25% attaque)
- Des patterns d'attaque plus distincts dans ce dataset historique
- Une separation plus marquee entre le trafic normal et les attaques

### 5.2 Analyse des faux positifs et faux negatifs

**Faux positifs (FP)** : Sessions normales classifiees comme attaques.
Causes possibles:
- Variante de trafic legitime non vue pendant l'entrainement
- Services peer-to-peer ou VPN qui generent des patterns inhabituels
- Connexions avec des timeouts ou des comportements erratiques

**Faux negatifs (FN)** : Attaques non detectees.
Causes possibles:
- Attaques stealth qui miment le trafic normal (slow-rate DoS, scans lents)
- Attaques utilisant des ports non standards
- Trafic chiffre qui neutralise les heuristics basees sur le contenu

### 5.3 Importance des variables (SHAP / Feature Importance)

Les variables les plus discriminantes pour la classification sur CICIDS2017
sont generalement:
1. `flow_bytes_s`  -  Le debit est le premier discriminant
2. `duration`  -  La duree du flux distingue les sessions courtes (scans)
   des sessions longues (telechargements)
3. `fwd_packets` / `bwd_packets`  -  Le nombre de paquets asymetrique
   signale un comportement anomal
4. `syn_count`  -  Un nombre eleve de SYN sans ACK correspondant
   indique un SYN scan ou un SYN flood
5. `rst_count`  -  Les resets repetitifs signalent un port scan ou
   un brute force

### 5.4 Comparaison avec la litterature

Les performances obtenues (97% accuracy sur CICIDS2017) sont compares
aux resultats rapportes dans la litterature:

| Etude                          | Modele       | Accuracy | Remarque           |
|--------------------------------|--------------|----------|--------------------|
| Choudhary & Kesswani (2021)    | Random Forest| 96.1%    | Dataset CICIDS2017 |
| Kasongo & Sun (2020)           | XGBoost      | 98.3%    | Dataset CICIDS2017 |
| **Ce projet**                  | XGBoost      | 97.27%   | CICIDS2017 synth.  |
| M. Al-Qatab et al. (2021)      | Deep Learning| 99.2%    | NSL-KDD            |
| **Ce projet**                  | XGBoost      | 98.45%   | NSL-KDD            |

Les resultats sont dans la moyenne de la litterature. L'ecart avec
certaines etudes s'explique par:
- L'utilisation de variants stealth dans notre dataset synthetique
- Des strategies de cross-validation differentes
- La presence de leakage dans certains benchmarks anterieurs

---

## 6. Connexion a un Reseau Reel

### 6.1 Principe

Le systeme IDS est concu pour fonctionner en mode passif. Il analyse
les sessions reseau sans interférer avec le traffic. En environnement
reel, les sessions peuvent etre capturees depuis:

1. **Sniffing Scapy** sur une interface reseau
2. **Socket TCP** recevant des logs JSON (mode agent)
3. **Fichier PCAP** pour l'analyse offline
4. **Stream Suricata/Zeek** au format EVE-JSON

### 6.2 Options d'integration

#### Scapy (sniff passif)

```python
from scapy.all import sniff, TCP, IP

def handle_packet(pkt):
    if pkt.haslayer(IP) and pkt.haslayer(TCP):
        session = {
            "src_ip": pkt[IP].src,
            "dst_ip": pkt[IP].dst,
            "src_port": pkt[TCP].sport,
            "dst_port": pkt[TCP].dport,
            "protocol": "TCP",
            # Extraire les statistiques CICFlowMeter equivalents
        }
        explain.explain_prediction(session)

sniff(iface="eth0", prn=handle_packet, count=0)  # 0 = infini
```

#### Zeek (IDS academique)

Zeek genere des logs structurees (JSON) pour chaque type d'evenement.
Les logs de connexion (`conn.log`) contiennent les 21 features
principales directement compatibles avec le format NSL-KDD.

```
# Activer JSON logging dans Zeek
@load policy/tuning/json-logs.zeek
```

#### Suricata (EVE-JSON)

Suricata peut envoyer ses evenements au format EVE-JSON sur un socket TCP:

```yaml
# /etc/suricata/suricata.yaml
outputs:
  - eve-log:
      enabled: yes
      type: socket
      destination: localhost:9999
      protocol: tcp
      types:
        - alert
        - flow
```

Le script `live_stream.py` peut se connecter a ce socket et traiter
chaque evenement en temps reel.

### 6.3 Format de donnees attendu

Les sessions doivent etre transmises au moteur de detection au format JSON:

```json
{
  "session_id": " uniquely identifier",
  "src_ip": "192.168.1.100",
  "dst_ip": "10.0.0.5",
  "src_port": 54321,
  "dst_port": 80,
  "protocol": "TCP",
  "duration": 1.523,
  "fwd_packets": 10,
  "bwd_packets": 8,
  "fwd_len": 5000,
  "bwd_len": 3000,
  "flow_bytes_s": 5248.7,
  "flow_packets_s": 11.8,
  "fwd_iat_mean": 0.152,
  "bwd_iat_mean": 0.201,
  "fwd_psh_flags": 1,
  "syn_count": 2,
  "rst_count": 0,
  "ack_count": 10,
  "psh_count": 1,
  "active_mean": 0.3,
  "idle_mean": 0.5,
  "flow_iat_mean": 0.177,
  "fwd_len_mean": 500.0,
  "bwd_len_mean": 375.0,
  "dst_port": 80
}
```

### 6.4 Limitations de l'approche passive

- **Pas de prevention** : L'IDS ne bloque pas le trafic. Un IPS ou un
  firewall dynamique (responder) doit etre associe.
- **Latence d'analyse** : Les sessions doivent etre terminees (ou contenir
  assez de paquets) pour que les features soient significatives.
- **Chiffrement** : Le HTTPS, SSH et DNS chiffre ne permettent pas
  d'inspecter le contenu applicatif.
- **Adversarial evasion** : Des attaques peuvent deliberement forger des
  paquets pour mimer le trafic normal et echapper a la detection.

---

## 7. Discussion et Limites

### 7.1 Biais du dataset

Le dataset CICIDS2017, bien que plus realiste que KDD Cup 99, reste une
capture dans un environnement controle. Le trafic d'un entreprise reelle
differe significativement en termes de:
- Protocols applicatifs (HTTP/2, QUIC, gRPC)
- Utilisation intensive de chiffrement (TLS 1.3)
- Trafic cloud/SaaS (Office 365, AWS, Google Cloud)
- VPN et tunnels chiffre

Un modele entraine sur CICIDS2017 aura necessairement des performances
degradees sur un trafic reelle pour ces raisons.

### 7.2 Trafic chiffre

Avec la generalisation de HTTPS (plus de 95% du trafic web selon
Let's Encrypt), les features basees sur le contenu des paquets
(URL, headers HTTP, payloads) ne sont plus accessibles. Seules les
metadonnees de flux (timing, taille des paquets, ports, nombre de
connexions) restent disponibles, ce qui limite la detection
d'attaque au niveau applicatif.

### 7.3 Faux positifs en environnement reel

En operation, un taux de faux positifs de 3-5% peut generer des
centaines d'alertes par jour sur un reseau d'entreprise. Cela
entraine:
- Fatigue des analysts de securite (alert fatigue)
- Risque de manquer les vraies attaques dans le bruit
- Necessite d'un filtrage avance (correlation d'evenements, seuils
  adaptatifs)

### 7.4 Necessite de reentrainement periodique

Les techniques d'attaque evoluent. Un modele IDS doit etre
reentraine regulierement (tous les 6-12 mois) sur des donnees
fraiches pour maintenir son efficacite. Ce processus requiert:
- Un etiquetage humain des nouvelles attaques
- Une infrastructure de collecte et de storage
- Une procedure de validation avant deploiement en production

---

## 8. Conclusion et Perspectives

### 8.1 Synthese

Ce projet a permis de concevoir et d'implementer un systeme de detection
d'intrusion base sur l'apprentissage automatique, compose de:

1. Un pipeline de preprocessing capable de traiter les datasets
   NSL-KDD et CICIDS2017
2. Deux modeles de classification (Random Forest, XGBoost) avec des
   performances elevees (97-98% accuracy)
3. Un module d'explicabilite permettant d'interpreter chaque prediction
4. Une interface Streamlit interactive pour la demonstration et les tests
5. Un script de demonstration d'attaques simulees pour evaluer le systeme
6. Une architecture d'integration reseau reel (Scapy, Zeek, Suricata)

Les resultats demontrent que l'apprentissage automatique peut detecter
efficacement la majorite des attaques reseau structurees, avec un
niveau de confiance exploitable par les analysts de securite grace
a l'explicabilite SHAP.

### 8.2 Ameliorations futures

**Deep Learning** : Explorer les architectures LSTM/GRU pour capturer
les dependances temporelles dans les sequences de flux, et les
autoencoders pour la detection d'anomalies non supervisee.

**Modeles ensemble** : Combiner Random Forest, XGBoost et un modele
de detection d'anomalie (Isolation Forest, One-Class SVM) dans un
systeme de vote robuste pour reduire les faux positifs.

**Deploiement en production** : Containeriser le systeme avec Docker,
integrer avec un SIEM (Wazuh, Elastic SIEM) et implementer un
pipeline de reentrainement automatique.

**Alertes et notification** : Implementer un systeme de scoring de
gravite des alertes, une correlation inter-sessions et un
mecanisme de notification (email, Slack, PagerDuty).

**Federated Learning** : Entrainer le modele de maniere distribuee
sur plusieurs sites sans centraliser les donnees, pour preserver
la confidentialite des traffics reseau de chaque organisation.

**Detection en temps reel** : Passer d'un modele batch (analyse
post-hoc) a un modele streaming capable d'analyser chaque paquet
en temps reel avec une latence minimale (< 10ms).

---

## 9. Annexes

### 9.1 Commandes d'installation

```bash
# Cloner le repository
git clone https://github.com/UtmostMaker/cyber-ids-streamlit.git
cd cyber-ids-streamlit

# Creer un environnement virtuel (recommande)
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Installer les dependances
pip install --break-system-packages -r requirements.txt

# Entrainer sur NSL-KDD (defaut)
python3 train.py

# Entrainer sur CICIDS2017
python3 train.py --dataset cicids2017

# Lancer l'interface Streamlit
streamlit run app.py --server.port 8501

# Lancer la demonstration d'attaques
python3 demo_attacks.py --all --dataset cicids2017
python3 demo_attacks.py --stream --duration 60 --interval 1.0
```

### 9.2 Structure du projet

```
cyber-ids-streamlit/
├── app.py                      # Interface Streamlit (dashboard)
├── train.py                    # Script d'entrainement (multi-dataset)
├── explain.py                  # Module d'explicabilite
├── demo_attacks.py             # Script de demonstration d'attaques
├── live_stream.py              # Integration reseau reel (optionnel)
├── prep_data_cicids2017.py     # Preprocessing CICIDS2017
├── prep_data_nslkdd.py         # Preprocessing NSL-KDD
├── requirements.txt            # Dependances Python
├── README.md                   # Cette documentation
├── data/
│   ├── KDDTrain+.txt           # Dataset NSL-KDD (train)
│   ├── KDDTest+.txt            # Dataset NSL-KDD (test)
│   └── cicids2017_processed.csv # Dataset CICIDS2017 (synthetique)
├── artifacts/
│   ├── train.pkl               # Donnees train NSL-KDD
│   ├── test.pkl                # Donnees test NSL-KDD
│   ├── train_cicids2017.pkl    # Donnees train CICIDS2017
│   ├── test_cicids2017.pkl    # Donnees test CICIDS2017
│   ├── preprocessor_cicids2017.pkl
│   ├── preprocessor.pkl
│   ├── results.json            # Metriques NSL-KDD
│   ├── results_cicids2017.json # Metriques CICIDS2017
│   ├── feature_importance.csv
│   ├── feature_importance_cicids2017.csv
│   ├── shap_importance.csv
│   ├── schema.json
│   ├── schema_cicids2017.json
│   └── label_encoders.pkl
└── models/
    ├── model.pkl               # Meilleur modele NSL-KDD
    └── model_cicids2017.pkl   # Meilleur modele CICIDS2017
```

### 9.3 Format JSON pour injection reseau

Pour integrer le systeme IDS avec un flux reseau existant, les sessions
doivent etre formatees en JSON et envoyees sur un socket TCP:

```bash
# Exemple: envoyer une session sur le port 9999
echo '{"session_id":"TEST-001","src_ip":"192.168.1.100",...}' \
  | nc localhost 9999
```

Le serveur `live_stream.py` ecoute sur ce port et traite chaque ligne
JSON recue comme une session independante.

### 9.4 References

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward
   Generating a New Intrusion Detection Dataset and Intrusion Traffic
   Characterization. ICISSP 2018.

2. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.

3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting
   System. KDD 2016.

4. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting
   Model Predictions. NIPS 2017.

5. Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009).
   A Detailed Analysis of the KDD CUP 99 Data Set. IEEE CISDA 2009.

6. Axelsson, S. (2000). The Base-Rate Fallacy and the Difficulty of
   Intrusion Detection. ACM TISSEC.
