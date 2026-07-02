import os
import csv
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="API d'analyse de sentiment")

# --- Configuration du chargement du modèle ---
# MODEL_SOURCE = "registry"   -> charge depuis le MLflow Model Registry (usage local, docker-compose up)
# MODEL_SOURCE = "pretrained" -> charge le modèle HuggingFace directement (usage CI/CD, cluster éphémère)
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "pretrained")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
REGISTERED_MODEL_NAME = "sentiment-model"
FALLBACK_CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"

classifier = None
model_status = {"source": None, "detail": None}


def load_model():
    """Charge le modèle selon MODEL_SOURCE, avec repli automatique en cas d'échec."""
    global classifier, model_status

    if MODEL_SOURCE == "registry":
        try:
            import mlflow

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
            classifier = mlflow.transformers.load_model(model_uri)
            model_status = {
                "source": "registry",
                "detail": f"{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}",
            }
            print(f"Modèle chargé depuis le Registry : {model_uri}")
            return
        except Exception as e:
            print(f"Échec du chargement depuis le Registry ({e}). Repli sur le modèle pré-entraîné.")

    # Mode "pretrained" ou repli après échec du Registry
    from transformers import pipeline

    classifier = pipeline("sentiment-analysis", model=FALLBACK_CHECKPOINT)
    model_status = {"source": "pretrained", "detail": FALLBACK_CHECKPOINT}
    print(f"Modèle chargé en mode pré-entraîné : {FALLBACK_CHECKPOINT}")


load_model()


class TextInput(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "API en ligne. Va sur /docs pour tester."}


@app.get("/health")
def health():
    """Utilisé par le CI/CD (kind/KServe) pour vérifier que le pod est prêt."""
    if classifier is None:
        return {"status": "error", "detail": "modèle non chargé"}, 503
    return {"status": "ok", "model": model_status}


@app.post("/predict")
def predict(input: TextInput):
    result = classifier(input.text)[0]
    with open("predictions_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [datetime.now().isoformat(), input.text, result["label"], result["score"]]
        )
    return {"label": result["label"], "score": round(result["score"], 4)}