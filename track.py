import mlflow
from transformers import pipeline

mlflow.set_experiment("sentiment-analysis-prototype")
classifier = pipeline("sentiment-analysis")

texts = [
    "I love this product!",
    "This is terrible.",
    "Pretty average experience."
]

with mlflow.start_run():
    mlflow.log_param("model_name", "distilbert-base-uncased-finetuned-sst-2-english")
    for i, text in enumerate(texts):
        result = classifier(text)[0]
        mlflow.log_metric(f"confidence_{i}", result["score"])
        mlflow.log_param(f"text_{i}_label", result["label"])

print("Run terminé.")