# Cyber IDS — Intrusion Detection System

Application Python de detection d'intrusions reseau basee sur des modeles de Machine Learning (Random Forest, XGBoost), avec un dashboard Streamlit interactif.

## Dataset

Dataset synthetique inspire de NSL-KDD, genere avec des distributions differentes pour les sessions normales et les attaques :

- **3000 sessions** (75% normal, 25% attaque)
- **20 features reseau** : duration, protocol_type, service, flag, src_bytes, dst_bytes, logged_in, serror_rate, etc.
- **Label** : `0` = normal, `1` = attaque

## Stack technique

- Python 3.12
- scikit-learn (Random Forest)
- XGBoost
- SHAP (explicabilite)
- Streamlit (dashboard)
- Pandas, NumPy, Matplotlib, Seaborn

## Installation

```bash
# Cloner le depot
git clone https://github.com/UtmostMaker/cyber-ids-streamlit.git
cd cyber-ids-streamlit

# Creer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# Installer les dependances
pip install -r requirements.txt
```

## Lancement

```bash
source .venv/bin/activate
streamlit run app.py --server.headless true
```

L'application sera disponible sur `http://localhost:8501`.

## Structure du projet

```
cyber-ids/
├── app.py               # Dashboard Streamlit (3 pages)
├── prep_data.py         # Generation + preprocessing
├── train.py             # Entrainement RF + XGBoost
├── explain.py           # Explicabilite (SHAP + fallback)
├── requirements.txt
├── README.md
├── data/
│   └── cybersecurity_intrusion_data.csv
├── models/
│   └── model.pkl        # Meilleur modele
├── artifacts/
│   ├── train.pkl / test.pkl
│   ├── label_encoders.pkl
│   ├── schema.json
│   ├── results.json
│   ├── feature_importance.csv
│   └── shap_importance.csv
└── logs/
    └── run_status.json
```

## Les 3 pages du dashboard

### 1. Modeles
Comparaison visuelle des modeles Random Forest et XGBoost :
- Metriques (Accuracy, F1, AUC-ROC)
- Bar chart comparatif
- Heatmaps des matrices de confusion
- Importance globale des variables

### 2. Simulation Temps Reel
- Bouton pour lancer une simulation de 20 sessions
- Affichage ligne par ligne avec badge rouge/vert et score de confiance
- Compteur d'attaques / sessions totales
- Theme visuel cyber

### 3. Tester une session
- Formulaire de saisie de toutes les features
- Prediction + confiance + barre de probabilites
- Top 3 facteurs explicatifs en francais
- Explication textuelle de la decision

## Metriques de performance

Les modeles atteignent des scores parfaits (100% F1, AUC-ROC) sur ce dataset synthetique, car les distributions normal/attaque sont deliberategment differentes. Dans un contexte reel, les performances seraient plus nuancées.

## Limites et ameliorations

- Dataset synthetique (pas de vraies attaques) — a remplacer par un jeu de donnees labelise reel (NSL-KDD, CICIDS2017)
- Pas de pipeline de monitoring en production
- Pas de retraining automatique
- La simulation temps reel est illustrative (pas de vrai flux reseau)
- SHAP peut etre lent sur de tres gros jeux de donnees
