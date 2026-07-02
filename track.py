"""
Entraîne/évalue le modèle de sentiment et l'enregistre dans le MLflow Model Registry.

Prérequis : le serveur MLflow doit tourner (docker-compose up -d), car le
Model Registry nécessite un backend store en base de données (Postgres),
pas le stockage fichier local par défaut.

Usage :
    docker-compose up -d
    python track.py
"""

import mlflow
import mlflow.transformers
from mlflow.tracking import MlflowClient
from transformers import pipeline

# --- Configuration ---
# SQLite est une vraie base de données : le Model Registry fonctionne,
# et aucun serveur MLflow n'a besoin d'être lancé au préalable.
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "sentiment-analysis-prototype"
REGISTERED_MODEL_NAME = "sentiment-model"
MODEL_CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

classifier = pipeline("sentiment-analysis", model=MODEL_CHECKPOINT)

texts = [
    "This is amazing!",
    "I am so disappointed.",
    "Not bad, could be better.",
    "Absolutely fantastic experience!",
    "Worst product ever.",
]

with mlflow.start_run() as run:
    # Parameters
    mlflow.log_param("model_name", MODEL_CHECKPOINT)
    mlflow.log_param("nombre_textes", len(texts))

    # Metrics
    scores = []
    for i, text in enumerate(texts):
        result = classifier(text)[0]
        mlflow.log_metric(f"confidence_{i}", result["score"])
        scores.append(result["score"])
        print(f"Text: {text} → {result['label']} ({result['score']:.4f})")

    score_moyen = sum(scores) / len(scores)
    mlflow.log_metric("score_moyen", score_moyen)

    # Artifact texte
    with open("resultats.txt", "w") as f:
        f.write(f"Score moyen: {score_moyen:.4f}\n")
        f.write(f"Nombre de textes: {len(texts)}\n")
    mlflow.log_artifact("resultats.txt")

    # --- Le modèle lui-même est loggé et enregistré dans le Registry ---
    model_info = mlflow.transformers.log_model(
        transformers_model=classifier,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    run_id = run.info.run_id

print(f"\nRun terminé. run_id = {run_id}")
print(f"Modèle enregistré dans le Registry sous le nom '{REGISTERED_MODEL_NAME}'.")

# Récupère automatiquement le numéro de version qui vient d'être créée
client = MlflowClient()
latest_versions = client.search_model_versions(f"run_id='{run_id}'")
if latest_versions:
    version = latest_versions[0].version
    print(f"Nouvelle version créée : {REGISTERED_MODEL_NAME} v{version}")
    print(f"Pour la promouvoir : python promote_model.py --version {version} --stage Staging")