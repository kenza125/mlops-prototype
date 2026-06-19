import pandas as pd

df = pd.read_csv("predictions_log.csv", names=["timestamp", "text", "label", "score"])
print("Nombre total de predictions :", len(df))
print(df["label"].value_counts())
print("Score moyen de confiance :", df["score"].mean())