from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import csv
from datetime import datetime

app = FastAPI(title="API d'analyse de sentiment")
classifier = pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "API en ligne. Va sur /docs pour tester."}

@app.post("/predict")
def predict(input: TextInput):
    result = classifier(input.text)[0]
    with open("predictions_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), input.text, result["label"], result["score"]])
    return {"label": result["label"], "score": round(result["score"], 4)}