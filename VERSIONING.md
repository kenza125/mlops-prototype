\# Stratégie de versioning — mlops-prototype



Ce projet versionne trois éléments distincts : le \*\*code\*\*, les \*\*données\*\* et le \*\*modèle\*\*.



\## 1. Versioning du code — Git tags



Chaque étape majeure du projet est marquée par un tag Git sémantique (`vX.Y.Z`) :



git tag v0.2.0

git push origin main --tags



| Tag | Description |

|-----|-------------|

| v0.1.0 | Prototype initial (API + CI de base) |

| v0.2.0 | Bloc 1 complet : MLflow SQLite + Model Registry + alias, DVC |



\## 2. Versioning des données — DVC



Le dataset d'évaluation (`data/eval\_reviews.csv`) est suivi par DVC, avec un remote \*\*local\*\* (pas de cloud, pas de carte bancaire nécessaire) :



dvc remote add -d localremote ../dvc-storage

dvc add data/eval\_reviews.csv

dvc push



Le fichier réel n'est jamais commité dans Git — seul `data/eval\_reviews.csv.dvc` (pointeur) l'est. Pour récupérer les données sur une autre machine :



dvc pull



Ce même dataset sera réutilisé au Bloc 3 comme référence pour la détection de drift (films → tweets).



\## 3. Versioning du modèle — MLflow Model Registry + alias



> \*\*Changement important\*\* : MLflow a déprécié les \*stages\* (`Staging` / `Production`) depuis la version 2.9, au profit d'un système d'\*\*alias\*\*, plus flexible. Ce projet utilise désormais exclusivement les alias.



\### Enregistrer une nouvelle version du modèle



python track.py



Ce script entraîne/évalue le modèle, logue les métriques dans MLflow (backend SQLite : `sqlite:///mlflow.db`), et enregistre automatiquement une nouvelle version dans le Model Registry sous le nom `sentiment-model`.



\### Promouvoir une version via un alias



Contrairement aux anciens stages, un alias peut être assigné à n'importe quelle version, et une même version peut avoir plusieurs alias.



python promote\_model.py --version 1 --alias production

python promote\_model.py --list

python promote\_model.py --remove-alias production



\### Charger un modèle par alias



Dans `api.py`, le modèle est chargé via l'URI d'alias :



model\_uri = "models:/sentiment-model@production"



\### Rollback



Le rollback consiste à réassigner l'alias `production` à une version antérieure :



python promote\_model.py --version <ancienne\_version> --alias production



Aucun redéploiement du code n'est nécessaire — au prochain chargement, l'API pointera automatiquement vers la version réassignée. Cette mécanique sert de base au rollback automatique prévu au Bloc 4.



\## Résumé des alias utilisés



| Alias | Rôle |

|-------|------|

| `production` | Version actuellement servie par l'API en mode `MODEL\_SOURCE=registry` |

