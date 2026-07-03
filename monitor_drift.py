"""
Detecte le data drift entre le dataset de reference (critiques de films)
et un nouveau dataset (tweets), en utilisant Evidently AI.

Usage :
    pip install evidently
    python monitor_drift.py
"""

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

REFERENCE_PATH = "data/eval_reviews.csv"
CURRENT_PATH = "data/tweets_reviews.csv"
OUTPUT_HTML = "drift_report.html"

reference_data = pd.read_csv(REFERENCE_PATH)
current_data = pd.read_csv(CURRENT_PATH)

print(f"Reference : {len(reference_data)} lignes ({REFERENCE_PATH})")
print(f"Actuel    : {len(current_data)} lignes ({CURRENT_PATH})")

for df in (reference_data, current_data):
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().apply(len)

report = Report(
    metrics=[
        DataDriftPreset(columns=["text_length", "word_count", "label"]),
    ]
)

my_eval = report.run(reference_data=reference_data, current_data=current_data)
my_eval.save_html(OUTPUT_HTML)

print(f"\nRapport de drift genere : {OUTPUT_HTML}")
print("Ouvre ce fichier dans un navigateur pour voir les resultats.")
