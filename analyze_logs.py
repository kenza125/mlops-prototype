"""
Analyse les logs de l'API pour detecter des anomalies (taux d'erreur eleve,
predictions incoherentes) et declenche un rollback automatique si besoin.

Usage :
    python analyze_logs.py
    python analyze_logs.py --auto-rollback
"""

import argparse
import subprocess
import sys
from collections import Counter

LOG_FILE = "predictions_log.csv"
# predictions_log.csv n'a pas de header, colonnes : timestamp,text,label,score
LOW_CONFIDENCE_THRESHOLD = 0.6  # score en dessous = prediction suspecte
ERROR_RATE_THRESHOLD = 0.3  # 30% de predictions a faible confiance declenche une alerte


def analyze_logs(log_file: str = LOG_FILE) -> dict:
    """Lit le fichier de log et calcule le taux de predictions a faible confiance."""
    import csv

    total = 0
    errors = 0

    try:
        with open(log_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                total += 1
                try:
                    score = float(row[3])
                except ValueError:
                    errors += 1  # score illisible = anomalie
                    continue
                if score < LOW_CONFIDENCE_THRESHOLD:
                    errors += 1
    except FileNotFoundError:
        print(f"Fichier de log introuvable : {log_file}")
        return {"total": 0, "errors": 0, "error_rate": 0.0}

    error_rate = errors / total if total > 0 else 0.0
    return {"total": total, "errors": errors, "error_rate": error_rate}


def get_previous_version(current_alias: str = "production") -> str | None:
    """Recupere la version precedente via mlflow (celle juste avant l'actuelle)."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    model_name = "sentiment-model"

    try:
        current = client.get_model_version_by_alias(model_name, current_alias)
        current_version = int(current.version)
    except Exception as e:
        print(f"Impossible de recuperer la version actuelle : {e}")
        return None

    if current_version <= 1:
        print("Pas de version anterieure disponible.")
        return None

    return str(current_version - 1)


def rollback(previous_version: str, alias: str = "production"):
    """Reassigne l'alias vers la version precedente."""
    print(f"Rollback en cours : {alias} -> version {previous_version}")
    result = subprocess.run(
        [sys.executable, "promote_model.py", "--version", previous_version, "--alias", alias],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Erreur lors du rollback : {result.stderr}")
        return False
    print("Rollback effectue avec succes.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-rollback", action="store_true", help="Effectue le rollback automatiquement si anomalie detectee")
    args = parser.parse_args()

    stats = analyze_logs()
    print(f"Total requetes analysees : {stats['total']}")
    print(f"Erreurs : {stats['errors']}")
    print(f"Taux d'erreur : {stats['error_rate']:.2%}")

    if stats["error_rate"] > ERROR_RATE_THRESHOLD:
        print(f"\nANOMALIE DETECTEE : taux d'erreur ({stats['error_rate']:.2%}) > seuil ({ERROR_RATE_THRESHOLD:.2%})")

        if args.auto_rollback:
            previous_version = get_previous_version()
            if previous_version:
                rollback(previous_version)
            else:
                print("Rollback impossible : pas de version anterieure trouvee.")
        else:
            print("Lance avec --auto-rollback pour corriger automatiquement.")
    else:
        print("\nAucune anomalie detectee. Systeme sain.")


if __name__ == "__main__":
    main()