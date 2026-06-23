# 🤖 MLOps Prototype — Sentiment Analysis Pipeline

> Prototype MLOps complet pour l'analyse de sentiment, intégrant FastAPI, Docker, MLflow et GitHub Actions CI/CD.

---

## 📌 Description

Ce projet est un **prototype MLOps** développé dans le cadre d'un apprentissage progressif des bonnes pratiques de déploiement de modèles de Machine Learning en production. Il implémente un pipeline complet d'analyse de sentiment basé sur un modèle **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`) via HuggingFace Transformers.

Le projet couvre les piliers fondamentaux du MLOps :
- **Serving** : API REST avec FastAPI
- **Containerisation** : Docker
- **Tracking d'expériences** : MLflow
- **CI/CD** : GitHub Actions
- **Logging** : Journalisation automatique des prédictions en CSV

---

## 🏗️ Architecture

```
mlops-prototype/
│
├── api.py               # API FastAPI — endpoint /predict
├── predict.py           # Prédiction standalone (script direct)
├── track.py             # Tracking MLflow des expériences
├── analyze_logs.py      # Analyse des logs de prédictions
├── test_api.py          # Tests automatisés de l'API
│
├── Dockerfile           # Image Docker (python:3.11-slim + uvicorn)
├── .dockerignore        # Fichiers exclus du build Docker
├── requirements.txt     # Dépendances Python
├── .gitignore           # Fichiers exclus de Git
│
└── .github/
    └── workflows/       # Pipelines CI/CD GitHub Actions
```

---

## 🛠️ Stack Technique

| Composant       | Technologie                         |
|-----------------|-------------------------------------|
| Framework API   | FastAPI + Uvicorn                   |
| Modèle NLP      | HuggingFace Transformers (DistilBERT) |
| Containerisation| Docker (python:3.11-slim)           |
| Experiment Tracking | MLflow                          |
| CI/CD           | GitHub Actions                      |
| Logging         | CSV (predictions_log.csv)           |
| Tests           | pytest / test_api.py                |

---

## 🚀 Démarrage Rapide

### Prérequis

- Python 3.11+
- Docker Desktop
- Git

### 1. Cloner le repo

```bash
git clone https://github.com/kenza125/mlops-prototype.git
cd mlops-prototype
```

### 2. Installation locale (sans Docker)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Lancer l'API localement

```bash
uvicorn api:app --reload --port 8000
```

Accéder à la documentation interactive : [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Tester l'API

```bash
# Via curl
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'

# Ou via le script de test
python test_api.py
```

**Exemple de réponse :**
```json
{
  "label": "POSITIVE",
  "score": 0.9998
}
```

---

## 🐳 Déploiement avec Docker

### Build de l'image

```bash
docker build -t mlops-sentiment .
```

### Lancer le conteneur

```bash
docker run -p 8000:8000 mlops-sentiment
```

L'API est disponible sur [http://localhost:8000](http://localhost:8000).

---

## 📊 Tracking MLflow

Pour enregistrer et visualiser les expériences :

```bash
# Lancer un run MLflow
python track.py

# Lancer l'interface MLflow UI
mlflow ui
```

Accéder à l'interface : [http://localhost:5000](http://localhost:5000)

Les métriques suivantes sont trackées :
- Paramètre du modèle utilisé (`model_name`)
- Score de confiance par prédiction (`confidence_i`)
- Label prédit par texte (`text_i_label`)

---

## 📝 Logs des Prédictions

Chaque appel à `/predict` est automatiquement journalisé dans `predictions_log.csv` avec :
- Horodatage (ISO 8601)
- Texte soumis
- Label prédit (`POSITIVE` / `NEGATIVE`)
- Score de confiance

---

## ⚙️ CI/CD avec GitHub Actions

Le pipeline CI/CD (`.github/workflows/`) s'exécute automatiquement à chaque push sur `main` et vérifie :
1. Installation des dépendances
2. Exécution des tests (`test_api.py`)
3. Build de l'image Docker

---

## 📁 Fichiers Clés

| Fichier           | Rôle |
|-------------------|------|
| `api.py`          | Serveur FastAPI avec endpoint POST `/predict` |
| `track.py`        | Script MLflow pour l'enregistrement d'expériences |
| `predict.py`      | Prédiction standalone sans API |
| `analyze_logs.py` | Analyse statistique des prédictions loggées |
| `test_api.py`     | Tests automatisés de l'API |
| `Dockerfile`      | Containerisation de l'application |

---

## 👩‍💻 Auteure

**Kenza** — Étudiante ingénieure en IA & Data Science, ENSTAB  
Spécialisation : Digitalisation et Analyse de Données (DAD)
