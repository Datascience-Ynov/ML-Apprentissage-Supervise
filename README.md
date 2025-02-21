# Machine-Learning: Apprentissage Supervisé

## Contexte du Projet

Ce projet vise à entraîner des modèles d'apprentissage supervisé sur le dataset **Fashion-MNIST**, un ensemble de données populaire pour la classification d'images. L'objectif est de relever les plus hautes performances possibles en comparant différentes approches de modélisation.

Le dataset **Fashion-MNIST** contient 70 000 images en niveaux de gris (28x28 pixels) réparties en 10 classes de vêtements et accessoires. Chaque image est associée à une étiquette correspondant à sa catégorie.

## Objectifs

1. **Préparation des données** : Transformer et normaliser les données pour les rendre exploitables par les modèles.
2. **Entraînement des modèles** : Tester plusieurs algorithmes d'apprentissage supervisé pour la classification.
3. **Optimisation des performances** : Utiliser des techniques comme la recherche par grille (Grid Search) pour optimiser les hyperparamètres.
4. **Comparaison des modèles** : Évaluer et comparer les performances des modèles pour identifier les meilleures approches.
5. **Visualisation des résultats** : Créer des visualisations pour mieux comprendre les performances et les prédictions.

## Technologies Utilisées

- **Langage** : Python
- **Bibliothèques** :
  - **Scikit-learn** : Pour l'implémentation des modèles d'apprentissage supervisé.
  - **Pandas** : Pour la manipulation des données.
  - **Matplotlib** : Pour la visualisation des données et des résultats.
  - **XGBoost** : Pour l'implémentation de modèles de boosting.
  - **Grid Search** : Pour l'optimisation des hyperparamètres.
- **Algorithmes Testés** :
  - **SVM (Support Vector Machines)**
  - **MLP (Multi-Layer Perceptron)**
  - **XGBoost**
  - **KNN (K-Nearest Neighbors)**
  - **Random Forest**
  - **Régression Linéaire**
  - **Régression Logistique**

## Étapes du Projet

### 1. Préparation des Données
- Chargement du dataset **Fashion-MNIST**.
- Normalisation des images (mise à l'échelle des valeurs des pixels entre 0 et 1).
- Division des données en ensembles d'entraînement et de test (60000 images pour l'entraînement et 10000 pour le test).

### 2. Entraînement des Modèles
- Implémentation des modèles suivants :
  - **SVM** : Pour la classification linéaire et non linéaire.
  - **MLP** : Un réseau de neurones simple pour la classification.
  - **XGBoost** : Un modèle de boosting basé sur les arbres de décision.
  - **KNN** : Un modèle basé sur la similarité des données.
  - **Random Forest** : Un ensemble d'arbres de décision pour améliorer la précision.
  - **Régression Linéaire** et **Régression Logistique** : Modèles de base pour la classification.

### 3. Optimisation des Hyperparamètres
- Utilisation de **Grid Search** pour optimiser les hyperparamètres des modèles.
- Recherche des meilleures combinaisons de paramètres pour maximiser la précision.

### 4. Évaluation des Modèles
- Calcul des métriques de performance :
  - **Précision (Accuracy)**
  - **Précision (Precision)**
  - **Rappel (Recall)**
  - **F1-Score**
  - **Matrice de Confusion**
- Comparaison des performances des modèles.

### 5. Visualisation des Résultats
- Visualisation des matrices de confusion pour chaque modèle.
- Graphiques comparatifs des performances (par exemple, barres pour l'accuracy).
- Visualisation des prédictions incorrectes pour comprendre les erreurs.

## Résultats Obtenus

- **Meilleur Modèle** : [Random Forest]
- **Comparaison des Performances** :
  - **SVM** : Précision de 88%
  - **MLP** : Précision de 88%
  - **XGBoost** : Précision de 89%
  - **KNN** : Précision de 88%
  - **Random Forest** : Précision de 91%
  - **Régression Logistique** : Précision de 85%

## Comment Utiliser ce Projet

1. **Cloner le dépôt** :
   ```bash
   git clone git@github.com:Datascience-Ynov/ML-Apprentissage-Supervise.git
   cd ML-Apprentissage-Supervise

2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt

3. **Exécuter le script principal** :
   ```bash
   python3 main.py


### Auteur
Nom : **AMOUSSA Mourad**

Contact : [mourad1.amoussa@outlook.com]

LinkedIn : [www.linkedin.com/in/mourad-amoussa]

GitHub : [https://github.com/Mourad2511]
