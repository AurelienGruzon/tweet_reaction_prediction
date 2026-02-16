# Tweet Reaction Prediction — Sentiment Analysis MLOps Project

## Contexte

Ce projet a été réalisé dans le cadre d’une mission visant à concevoir un système complet d’analyse de sentiments sur des tweets.  
L’objectif est de détecter automatiquement les messages exprimant un mécontentement client afin de faciliter leur priorisation dans un contexte de relation client.

Le projet couvre l’ensemble du cycle de vie d’un modèle de Machine Learning, de l’expérimentation à la mise en production, dans une démarche MLOps.

---

## Objectifs

- Comparer plusieurs approches de modélisation :
  - modèle classique (TF-IDF + régression logistique)
  - modèle Deep Learning (LSTM + embeddings)
  - modèle Transformer (DistilBERT)
- Mettre en place un tracking des expérimentations avec MLflow
- Déployer un modèle via une API
- Implémenter une interface de test
- Mettre en place un monitoring en production

---

## Modèles implémentés

### Baseline
TF-IDF + Logistic Regression  
Modèle de référence pour comparer les approches plus complexes  

### Deep Learning
Réseau LSTM bidirectionnel  
Comparaison stemming vs lemmatization  
Comparaison Word2Vec vs FastText  

### Modèle avancé
DistilBERT fine-tuned  
Modèle retenu pour la mise en production  

---

## Tracking des expérimentations

Les expérimentations sont suivies avec MLflow :

- hyperparamètres
- métriques (accuracy, F1, AUC)
- temps d’entraînement et de prédiction
- artefacts et modèles

Des captures de l’interface MLflow sont disponibles dans le dossier docs/.

---

## Architecture du projet

notebooks/ : EDA et modélisation  
src/ : code source (API, modèle, utils)  
models/ : modèles sauvegardés  
tests/ : tests unitaires  
ui_streamlit/ : interface de test  
docs/ : article et preuves  
.github/workflows : pipeline CI/CD  
Dockerfile : conteneurisation  
pyproject.toml : dépendances  

---

## API de prédiction

Endpoint :

POST /predict

Exemple de requête :

{
  "text": "I hate this airline"
}

Réponse :

{
  "label": "negative",
  "proba_negative": 0.99
}

---

## Lancer le projet en local

Installer les dépendances :

poetry install

Lancer l’API :

uvicorn src.api.app:app --reload

Lancer l’interface Streamlit :

streamlit run ui_streamlit/app.py

---

## Déploiement

Le projet est conteneurisé avec Docker et déployé automatiquement via GitHub Actions.

Pipeline :
- exécution des tests unitaires
- build de l’image Docker
- push sur registry
- déploiement sur Azure WebApp

---

## Monitoring

Le suivi en production est assuré avec Azure Application Insights :

- traces sur prédictions incorrectes
- déclenchement d’alertes
- suivi de la latence et du volume de requêtes

---

## Tests unitaires

pytest

---

## Dépendances principales

FastAPI  
Transformers  
PyTorch  
MLflow  
Streamlit  

La liste complète est disponible dans pyproject.toml.

---

## Contenu du dossier docs

Article de blog détaillant la démarche  
Captures MLflow  
Captures CI/CD  
Captures Azure Monitoring  

---

## Démarche MLOps mise en œuvre

tracking des expérimentations  
reproductibilité  
gestion de versions  
tests automatisés  
déploiement continu  
monitoring en production  

---

## Auteur

Projet réalisé par Aurélien Gruzon dans le cadre d’un parcours Data Scientist.
