# MLOps Prototype — Sentiment Analysis API

Pipeline MLOps complet pour une API d'analyse de sentiment (HuggingFace DistilBERT + FastAPI), couvrant le versioning, le déploiement automatisé, le monitoring de drift et le rollback automatique.

## Architecture

```
mlops-prototype/
├── api.py                  # API FastAPI (mode registry ou pretrained)
├── track.py                 # Entraînement + enregistrement MLflow
├── promote_model.py         # Gestion des alias MLflow (production, etc.)
├── analyze_logs.py          # Détection d'anomalies + rollback automatique
├── monitor_drift.py         # Détection de data drift (Evidently AI)
├── predict.py
├── test_api.py
├── data/
│   ├── eval_reviews.csv         # Dataset de référence (critiques de films, DVC)
│   └── tweets_reviews.csv       # Dataset "actuel" simulé (tweets, pour le drift)
├── .github/workflows/ci-cd.yml  # Pipeline CI/CD
├── Dockerfile
├── mlflow.db                 # Backend MLflow SQLite (non versionné)
├── VERSIONING.md
└── README.md
```

## Les 4 blocs

### Bloc 1 — Versioning
Trois éléments versionnés indépendamment :
- **Code** : Git + tags sémantiques (`v0.2.0`, `v0.3.0`, ...)
- **Données** : DVC, avec un remote **local** (`../dvc-storage`, aucun cloud requis)
- **Modèle** : MLflow Model Registry, backend SQLite (`sqlite:///mlflow.db`), versions gérées par **alias** (`@production`) plutôt que par les anciens *stages* dépréciés depuis MLflow 2.9

Détails complets : voir [VERSIONING.md](./VERSIONING.md)

### Bloc 2 — CI/CD déploiement automatisé
À chaque push sur `main`, `.github/workflows/ci-cd.yml` exécute trois jobs en cascade :
1. `test` — tests unitaires (`pytest`)
2. `build-and-push` — build de l'image Docker, push vers Docker Hub (`kenza125/sentiment-api:latest`)
3. `deploy` — déploiement sur un cluster **kind** éphémère créé dans le runner GitHub Actions, avec **KServe en mode RawDeployment** (léger, sans Istio/Knative), puis vérification de santé (`/health`)

Aucune infrastructure persistante, aucun coût cloud.

### Bloc 3 — Monitoring & Data Drift
`monitor_drift.py` utilise **Evidently AI** pour comparer :
- le dataset de référence (`data/eval_reviews.csv`, critiques de films)
- un dataset "actuel" simulé (`data/tweets_reviews.csv`, style tweets)

Un rapport HTML (`drift_report.html`) est généré, mesurant le drift sur la longueur des textes, le nombre de mots et la distribution des labels.

```bash
python monitor_drift.py
```

### Bloc 4 — Debug automatique & Rollback
`analyze_logs.py` analyse `predictions_log.csv` (colonnes : `timestamp, text, label, score`) et détecte une anomalie lorsque le taux de prédictions à faible confiance (score < 0.6) dépasse 30 %.

```bash
python analyze_logs.py                # détection seule
python analyze_logs.py --auto-rollback  # détection + rollback automatique
```

En cas d'anomalie, le script réassigne automatiquement l'alias `production` à la version précédente du modèle via `promote_model.py` — aucun redéploiement de code nécessaire, seule l'étiquette MLflow bouge.

**Scénario testé et validé** : promotion d'une version dégradée (v2) → détection automatique (taux d'erreur 30.77 % > seuil 30 %) → rollback automatique vers v1 → vérification.

## Installation

```bash
git clone https://github.com/kenza125/mlops-prototype.git
cd mlops-prototype
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
dvc pull                          # récupère le dataset versionné
```

## Utilisation locale

```bash
# Entraîner et enregistrer une version du modèle
python track.py

# Promouvoir une version en production
python promote_model.py --version 1 --alias production
python promote_model.py --list

# Lancer l'API (mode registry)
$env:MODEL_SOURCE="registry"
uvicorn api:app --reload

# Tester
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{\"text\": \"I love this movie\"}'
```

## Stack technique

| Composant | Outil |
|---|---|
| API | FastAPI + Uvicorn |
| Modèle | HuggingFace Transformers (DistilBERT) |
| Tracking / Registry | MLflow (backend SQLite) |
| Versioning données | DVC (remote local) |
| CI/CD | GitHub Actions |
| Déploiement | kind + KServe (RawDeployment) |
| Monitoring drift | Evidently AI |
| Conteneurisation | Docker |

## Historique des versions

| Tag | Contenu |
|---|---|
| v0.2.0 | Bloc 1 : versioning complet (Git, DVC, MLflow alias) |
| v0.3.0 | Bloc 2 : CI/CD déploiement automatisé (kind + KServe) |
| v0.4.0 | Bloc 3 : monitoring & data drift (Evidently AI) |
| v0.5.0 | Bloc 4 : détection d'anomalies & rollback automatique |
| v1.0.0 | Projet complet, les 4 blocs validés |
