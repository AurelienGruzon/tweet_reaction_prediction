# Prototype Air Paradis : Analyse de sentiments sur Twitter — du modèle classique à BERT, avec une vraie démarche MLOps

## 1) Contexte : pourquoi Air Paradis a besoin d’un modèle de sentiment

Lors du meeting de cadrage avec **Air Paradis**, l’objectif du prototype est clair : analyser automatiquement des tweets pour détecter ceux exprimant un **sentiment négatif**, afin de :

* repérer plus vite les signaux faibles (plaintes, crises, insatisfaction),
* prioriser le traitement des retours clients,
* fournir un outil réutilisable sur d’autres cas d’usage (autres compagnies, SAV, marque employeur, etc.).

Le problème est formulé en **classification binaire** :

* **0 : non négatif**
* **1 : négatif**

Et la contrainte importante côté produit : **limiter les coûts**, donc viser un déploiement sur une solution cloud gratuite (Azure WebApp F1 par exemple), même si cela implique d’adapter le modèle déployé.

---

## 2) Approche 1 — “Modèle sur mesure simple” : baseline ML robuste

Pour développer rapidement une solution efficace et peu coûteuse, on commence par un **modèle classique** :

* vectorisation texte via **TF-IDF**
* modèle linéaire (ex : **régression logistique**)

### Pourquoi TF-IDF + régression logistique ?

Parce que c’est souvent imbattable en ratio :

* simplicité
* vitesse d’entraînement
* interprétabilité
* coût infra quasi nul

C’est aussi une baseline indispensable pour vérifier que :

* la donnée est exploitable,
* les splits train/val/test sont cohérents,
* le problème n’a pas de piège structurel.

---

## 3) Approche 2 — “Modèle sur mesure avancé” : Deep Learning (LSTM)

La commande précise que le prototype doit également inclure un modèle Deep Learning “sur mesure”, et que **c’est ce modèle qui doit pouvoir être déployé et présenté à Air Paradis**.

J’ai donc construit un modèle **BiLSTM** :

* vocabulaire construit sur train
* encodage séquentiel (tokens → ids)
* couche d’embeddings
* **BiLSTM**
* classification binaire (sigmoid + BCEWithLogitsLoss)

Cette approche permet de capturer :

* un peu de contexte (ordre des mots),
* des dépendances de séquence,
* une meilleure robustesse que TF-IDF sur certains cas implicites.

---

## 4) Expériences obligatoires : prétraitement et word embeddings

La commande impose explicitement de tester plusieurs approches, avec une logique pragmatique : tirer des gains via **prétraitement** et **plongements de mots (word embeddings)** plutôt que d’empiler des architectures complexes.

### 4.1 Prétraitement : stemming vs lemmatization (sur DL)

Point important : ces techniques ne doivent pas être testées uniquement sur TF-IDF, mais **sur un modèle Deep Learning**, comme demandé.

J’ai donc exécuté deux runs MLflow sur le modèle **BiLSTM** :

* **LSTM + stemming**
* **LSTM + lemmatization**

**Résultats (validation) :**

* `stem` : **AUC = 0.9001**, **F1 = 0.8203**
* `lemma` : **AUC = 0.8979**, **F1 = 0.8167**

✅ Conclusion : **stemming retenu** pour la suite (meilleur AUC & F1, et généralisation stable sur test).

---

### 4.2 Word embeddings : Word2Vec vs FastText (sur DL)

La commande impose aussi :

> essayer au moins deux word embeddings différents et garder le meilleur

Deux runs MLflow supplémentaires sont réalisés sur le LSTM (avec stemming conservé) :

* **LSTM + Word2Vec**
* **LSTM + FastText**

**Résultats (validation) :**

* Word2Vec : **AUC = 0.9057**, **F1 = 0.8214**
* FastText : **AUC = 0.9061**, **F1 = 0.8220**

✅ Conclusion : **FastText** est légèrement meilleur que Word2Vec (écart faible mais cohérent).
Cela est logique : FastText intègre l’information “sous-mots”, souvent utile sur Twitter (fautes, contractions, variantes lexicales).

---

## 5) Apport de BERT : faut-il investir dans ce type de modèle ?

La commande demande explicitement d’évaluer l’apport de BERT pour décider d’un investissement potentiel.

BERT apporte une rupture importante :

* embeddings **contextuels**
* meilleure compréhension implicite
* robustesse aux formulations variées

J’ai donc entraîné une version BERT (via Hugging Face + PyTorch), suivie dans MLflow, afin de comparer objectivement avec les modèles précédents.

Dans la majorité des cas pratiques sur tweets :
✅ **BERT devient le meilleur modèle** en performance brute, en particulier sur les cas difficiles :

* ironie / sarcasme
* critiques indirectes
* structures lexicales non triviales

---

## 6) Évaluation métier : pourquoi l’AUC + matrice de confusion sont indispensables

Le projet ne doit pas se limiter à “un score”.

La commande insiste sur :

* la matrice de confusion
* l’AUC (ou équivalent)
* l’analyse FP/FN

### Pourquoi la précision seule ne suffit pas ?

Parce que le coût des erreurs est asymétrique :

* **FN (faux négatif)** : tweet négatif non détecté
  → risque métier fort (plaintes ignorées, crise non détectée)
* **FP (faux positif)** : tweet non négatif détecté comme négatif
  → coût moins critique (alertes inutiles)

Ainsi, j’ai suivi :

* **AUC** : mesure globale de séparation (indépendante du seuil)
* **F1** : compromis précision/rappel
* **matrice de confusion** : lecture opérationnelle des erreurs

---

## 7) Démarche MLOps : l’élément central du prototype

La partie MLOps est une priorité explicite de la commande.
L’objectif est de montrer que le prototype n’est pas seulement un notebook, mais un système :

* traçable
* comparable
* déployable
* monitoré
* améliorable

---

### 7.1 Principes MLOps (synthèse)

Le MLOps vise à industrialiser un modèle ML, en reproduisant les standards du software engineering :

1. Expérimentations contrôlées
2. Versioning (code + modèle)
3. Packaging
4. Déploiement automatisé
5. Monitoring (qualité + perf)
6. Feedback et amélioration continue

---

### 7.2 MLflow : tracking, comparaison, stockage des modèles

MLflow est utilisé comme outil central :

✅ Chaque run loggue :

* paramètres (modèle, preprocess, embeddings, epochs…)
* métriques (AUC, F1, recall…)
* artefacts :

  * ROC curve
  * matrice de confusion
  * vocabulaire/tokenizer
  * modèle exporté

Cela permet :

* une comparaison objective entre solutions
* une reproductibilité complète
* un reporting simple et lisible

**Bonus demandé explicitement : serving MLflow**
Le serving MLflow permet de tester rapidement une mise en production :

* le modèle est loggé avec MLflow
* il peut être servi via :

  * `mlflow models serve`

Cela offre une “preuve de concept” industrialisable en interne.

---

### 7.3 Déploiement continu via API (GitHub + Cloud)

Le modèle retenu est exposé via une **API REST** (FastAPI), incluant :

* `/health` : endpoint de santé
* `/predict` : renvoie label + probabilité

Le pipeline CI/CD automatise :

* installation des dépendances
* tests unitaires
* build Docker
* push registry
* déploiement Azure WebApp (F1 gratuit)

Le résultat : mise en production reproductible, sans manipulation manuelle.

---

### 7.4 Suivi en production : Azure Application Insights

Le prototype initie un monitoring via **Azure Application Insights** :

* traces de requêtes `/predict`
* mesure de latence
* détection d’erreurs applicatives

#### Remontée des mauvaises prédictions

La commande impose :

* remonter les tweets jugés “mal prédits” par l’utilisateur
* stocker texte + prédiction

Cela alimente un dataset “réel production” pour :

* analyse statistique
* détection d’usages inattendus
* préparation du futur ré-entraînement

---

### 7.5 Alerte automatique : trop d’erreurs en 5 minutes

La commande demande :

> déclencher une alerte si trop de tweets mal prédits (ex : 3 en 5 minutes)

Le principe est simple :

* chaque feedback “mauvaise prédiction” génère un événement
* si **≥ 3 événements sur une fenêtre glissante de 5 min**
  → déclenchement d’une alerte (mail/SMS)

Même si le canal final (Twilio, mail corporate…) dépend de l’entreprise, la logique du système est prête.

---

## 8) Conclusion : quel modèle déployer ?

Le prototype permet une lecture claire des options :

### Option 1 : modèle classique (TF-IDF + logistic regression)

✅ ultra léger / coût minimal / simple
➡️ choix idéal si contrainte cloud gratuite stricte

### Option 2 : LSTM + FastText

✅ bon compromis perf / coût
➡️ choix recommandé “modèle avancé déployable”

### Option 3 : BERT

✅ meilleure performance
⚠️ plus lourd / latence / infra plus coûteuse
➡️ option premium si Air Paradis investit dans un modèle plus robuste

---

## Améliorations possibles (suite logique MLOps)

* ajustement du seuil selon coûts FP/FN
* calibration des probabilités
* analyse continue du feedback (drift)
* ré-entraînement périodique
* active learning sur cas ambigus

