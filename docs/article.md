# Prototype Air Paradis : analyse de sentiments sur Twitter — du modèle classique à BERT, avec une démarche MLOps complète

## 1) Contexte : pourquoi Air Paradis a besoin d’un modèle de sentiment

Lors du meeting de cadrage avec Air Paradis, l’objectif du prototype est clairement défini : **analyser automatiquement des tweets afin d’identifier ceux exprimant un sentiment négatif**.
Dans un contexte où les compagnies aériennes sont fortement exposées sur les réseaux sociaux, Twitter constitue un canal critique pour la perception de la marque.

L’enjeu métier est multiple :

* détecter rapidement des signaux faibles (plaintes émergentes, insatisfaction diffuse),
* prioriser le traitement des retours clients,
* anticiper des situations de crise avant qu’elles ne prennent de l’ampleur,
* disposer d’un outil réutilisable sur d’autres cas d’usage (autres compagnies, service client, marque employeur, analyse d’e-réputation).

Une veille humaine manuelle est coûteuse, peu scalable et réactive avec retard. L’automatisation permet au contraire de traiter **un flux massif et continu**, avec un coût marginal faible une fois le modèle déployé.

Le problème est formulé comme une **classification binaire** :

* 0 : non négatif
* 1 : négatif

Une contrainte produit majeure est posée dès le départ : **limiter les coûts d’infrastructure**. Le prototype doit donc être compatible avec un déploiement sur une solution cloud gratuite (par exemple Azure WebApp F1), quitte à adapter le modèle effectivement mis en production.

---

## 2) Approche 1 — “Modèle sur mesure simple” : une baseline ML robuste

La première étape consiste à établir une baseline simple, robuste et interprétable :

* vectorisation du texte via TF-IDF,
* modèle linéaire de type régression logistique.

Ce choix est volontairement pragmatique. TF-IDF combiné à un modèle linéaire offre souvent un excellent compromis entre :

* simplicité de mise en œuvre,
* rapidité d’entraînement,
* interprétabilité (poids des termes),
* coût infrastructure quasi nul.

Cette approche permet de valider rapidement plusieurs points essentiels :

* la qualité intrinsèque des données,
* la cohérence des splits train / validation / test,
* l’absence de pièges structurels (déséquilibre extrême, bruit excessif).

Cette baseline joue également un rôle clé : **servir de point de comparaison objectif** pour évaluer l’apport réel de modèles plus complexes.

---

## 3) Approche 2 — “Modèle sur mesure avancé” : Deep Learning avec LSTM

La commande impose explicitement l’intégration d’un modèle Deep Learning “sur mesure”, capable d’être déployé et présenté à Air Paradis.

Un modèle **BiLSTM** est donc conçu avec l’architecture suivante :

* construction du vocabulaire sur le jeu d’entraînement,
* encodage séquentiel (tokens → identifiants),
* couche d’embeddings,
* BiLSTM bidirectionnel,
* couche de classification binaire avec sigmoid et BCEWithLogitsLoss.

Cette architecture permet de capturer :

* l’ordre des mots,
* certaines dépendances de séquence,
* des formes de contexte absentes des modèles TF-IDF.

Le modèle reste volontairement maîtrisé en taille afin de conserver une **latence compatible avec une API déployée sur infrastructure gratuite**, sous réserve d’un vocabulaire et d’embeddings contrôlés.

---

## 4) Expérimentations obligatoires : prétraitement et word embeddings

La commande insiste sur une démarche pragmatique : **obtenir des gains de performance via le prétraitement et les embeddings**, plutôt que par l’empilement d’architectures complexes.

### 4.1 Prétraitement : stemming vs lemmatization (sur Deep Learning)

Deux expérimentations MLflow sont réalisées sur le modèle BiLSTM :

* LSTM + stemming
* LSTM + lemmatization

Résultats (validation) :

* stemming : AUC = 0.9001, F1 = 0.8203
* lemmatization : AUC = 0.8979, F1 = 0.8167

Le stemming est retenu. Sur Twitter, où les variations lexicales sont fréquentes, cette approche plus agressive permet une meilleure généralisation et une stabilité accrue sur le jeu de test.

### 4.2 Word embeddings : Word2Vec vs FastText

Deux nouvelles expérimentations MLflow sont menées, avec le stemming conservé :

* LSTM + Word2Vec
* LSTM + FastText

Résultats (validation) :

* Word2Vec : AUC = 0.9057, F1 = 0.8214
* FastText : AUC = 0.9061, F1 = 0.8220

FastText est retenu. Son utilisation des sous-mots est particulièrement adaptée aux tweets, où fautes, abréviations et variantes orthographiques sont fréquentes.

---

## 5) Apport de BERT : faut-il investir dans ce type de modèle ?

La commande demande explicitement d’évaluer l’intérêt de BERT afin d’éclairer une décision d’investissement potentiel.

BERT apporte une rupture technologique majeure :

* embeddings contextuels,
* meilleure compréhension implicite,
* robustesse accrue face à l’ambiguïté lexicale.

Une version BERT est entraînée via Hugging Face et PyTorch, suivie dans MLflow.
Les résultats montrent que **BERT devient le meilleur modèle en performance brute**, notamment sur :

* l’ironie et le sarcasme,
* les critiques indirectes,
* les formulations complexes.

Cependant, ces gains s’accompagnent de contraintes fortes :

* latence plus élevée,
* consommation mémoire importante,
* incompatibilité avec une infrastructure cloud gratuite.

BERT est donc positionné comme une **option premium**, pertinente si Air Paradis accepte un investissement infrastructure plus conséquent.

---

## 6) Évaluation métier : pourquoi l’AUC et la matrice de confusion sont indispensables

La précision seule est insuffisante pour ce cas d’usage. Le coût des erreurs est asymétrique :

* faux négatif : tweet négatif non détecté, avec un risque métier élevé (plainte ignorée, crise non anticipée),
* faux positif : alerte inutile, coût opérationnel plus faible.

L’évaluation repose donc sur :

* l’AUC, mesure globale de séparation indépendante du seuil,
* le F1-score, compromis précision / rappel,
* la matrice de confusion, lecture opérationnelle des erreurs.

L’analyse qualitative des erreurs montre que les faux négatifs concernent souvent des formulations implicites ou ironiques, ce qui justifie l’intérêt de modèles contextuels ou d’un ajustement fin du seuil de décision.

---

## 7) Démarche MLOps : industrialiser le prototype

La dimension MLOps constitue le cœur du projet. L’objectif n’est pas de produire un simple notebook, mais un **système industrialisable**, traçable et améliorable dans le temps.

### 7.1 Pourquoi une démarche MLOps est indispensable

Un modèle de Machine Learning n’a de valeur que s’il peut être :

* reproduit,
* comparé,
* déployé,
* observé,
* amélioré.

Sans démarche MLOps, un projet reste fragile : dépendant d’un environnement local, difficile à maintenir et impossible à auditer.

### 7.2 Principes MLOps appliqués

Les principes suivants sont explicitement mis en œuvre :

**Reproductibilité**
Chaque expérimentation peut être rejouée à l’identique grâce au versioning du code, au tracking MLflow et à la maîtrise de l’environnement (Poetry, Docker).

**Traçabilité**
Chaque modèle est associé à un run MLflow, incluant paramètres, métriques et artefacts. Il est possible d’expliquer précisément quel modèle est en production et comment il a été entraîné.

**Comparabilité**
Les modèles sont comparés sur des splits identiques, avec des métriques homogènes, via l’interface MLflow.

**Séparation entraînement / inférence**
L’entraînement est réalisé offline (notebooks, scripts). L’API ne contient que de l’inférence, garantissant stabilité et performance.

**Automatisation**
Les tests unitaires, le build Docker et le déploiement sont automatisés via un pipeline CI/CD.

**Observabilité**
Le modèle en production est instrumenté : latence, erreurs et requêtes sont tracées.

**Boucle de feedback**
Les retours utilisateurs sont collectés pour alimenter une amélioration continue.

### 7.3 Implémentation concrète dans le prototype Air Paradis

* MLflow est utilisé pour le tracking, la comparaison et le stockage des modèles.
* Git assure le versioning du code.
* Docker garantit la portabilité de l’API.
* GitHub Actions automatise les tests et le déploiement.
* Azure WebApp héberge l’API sur une infrastructure gratuite.

### 7.4 Suivi en production et feedback utilisateur

Azure Application Insights permet de suivre :

* le volume de requêtes,
* la latence,
* les erreurs applicatives.

Les tweets jugés mal prédits par l’utilisateur sont remontés et stockés, constituant un **jeu de données réel de production**, exploitable pour :

* l’analyse statistique,
* la détection de dérive,
* la préparation d’un futur ré-entraînement.

### 7.5 Alerte automatique

Une alerte est déclenchée lorsqu’un nombre anormal de mauvaises prédictions est détecté sur une fenêtre temporelle donnée.
Ce mécanisme démontre la capacité du système à réagir en production, condition essentielle pour une exploitation industrielle.

---

## 8) Conclusion : quel modèle déployer ?

Trois options claires émergent :

* **TF-IDF + régression logistique** : ultra léger, coût minimal, idéal sous contrainte stricte.
* **LSTM + FastText** : meilleur compromis performance / coût, recommandé pour un déploiement avancé.
* **BERT** : performance maximale, mais exigences infrastructure élevées.

Le prototype fournit ainsi à Air Paradis **une aide à la décision claire**, fondée sur des critères techniques, métiers et opérationnels, tout en démontrant une démarche MLOps complète et industrialisable.