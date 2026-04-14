# Prédiction de la Gravité des Accidents de la Route
### Projet Machine Learning — RTA Dataset | Addis Abeba, Éthiopie

---

> J'ai choisi ce sujet parce que la question posée est concrète : si on peut prédire à l'avance la gravité d'un accident à partir des circonstances observées, les services d'urgence peuvent intervenir plus rapidement et de façon mieux ciblée. C'est un problème où le Machine Learning a une valeur réelle.

---

## Auteur

| | |
|--|--|
| **Nom** | Rania Shili|
| **Module** | Machine Learning |
| **Encadrant** | M. Abdallah Khemais |
| **Année** | 2025 – 2026 |

---

## Problématique

> **Peut-on prédire automatiquement la gravité d'un accident de la route à partir des conditions au moment des faits ?**

La variable cible est `Accident_severity`, avec trois classes :
- `Slight Injury` — blessure légère
- `Serious Injury` — blessure grave
- `Fatal Injury` — décès

Il s'agit d'une tâche de **classification multiclasse supervisée**.

---

## Dataset

Le **RTA Dataset** (Road Traffic Accidents) est issu d'enregistrements réels collectés sur 8 ans par la police d'Addis Abeba, Éthiopie.

| Attribut | Valeur |
|----------|--------|
| Source | Kaggle / UCI Machine Learning Repository |
| Volume total | ~12 181 enregistrements |
| Volume de travail | **3 000 lignes** (shuffle aléatoire complet, `random_state=42`) |
| Dimensions | 3 000 lignes × 32 colonnes |
| Variables numériques | 2 (`Number_of_vehicles_involved`, `Number_of_casualties`) |
| Variables catégorielles | 30 |
| Variable cible | `Accident_severity` (3 classes) |

### Pourquoi ce dataset ?
Il dépasse le seuil requis de 500 lignes, propose un problème métier concret, et présente un défi méthodologique réel : un fort déséquilibre entre les classes (Slight Injury majoritaire à ~83%), ce qui oblige à aller au-delà de la simple accuracy et à utiliser des métriques adaptées.

---

## Structure du Projet

```
PROJET_ML_RTA/
│
├── Data/
│   ├── RTA Dataset.csv              # Dataset original complet
│   └── df_eda_ready.csv             # Données préparées après EDA
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Brouillon EDA 
│   ├── 01_EDA_final.ipynb           # EDA final soigné et commenté
│   ├── 02_Modeling.ipynb            # Brouillon modélisation 
│   └── 02_Modeling_final.ipynb      # Modélisation finale complète
│
├── src/
│   ├── best_model_rf.pkl            # Modèle Random Forest sérialisé
│   └── label_encoder.pkl            # Encodeur de la variable cible
│
├── app.py                           # Application Streamlit 
├── requirements.txt
└── README.md
```

---

## Démarche Méthodologique

### Chargement et Échantillonnage
Le shuffle est appliqué sur les **12 181 lignes complètes** du dataset avant tout échantillonnage. On extrait ensuite les 3 000 premières lignes. Cette approche garantit un échantillon représentatif sans biais d'ordre.

```python
df_raw = pd.read_csv('RTA Dataset.csv')
df = df_raw.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.head(3000)
```

### Valeurs Manquantes — Imputation, pas Suppression
Sur les 32 colonnes, **16 contiennent des valeurs manquantes**. Les deux colonnes les plus touchées sont `Defect_of_vehicle` (34,63%) et `Service_year_of_vehicle` (32,07%). Ces valeurs ne sont **jamais supprimées** : elles sont traitées par imputation dans le pipeline sklearn, ce qui préserve toute l'information disponible dans les autres colonnes de chaque ligne.

### Déséquilibre des Classes
La classe `Slight Injury` est largement majoritaire (~83%). Un modèle naïf qui prédit toujours cette classe atteindrait une accuracy élevée sans rien apprendre. Pour cette raison, `class_weight='balanced'` est activé sur tous les modèles, et les métriques d'évaluation vont au-delà de la simple accuracy.

### Pipeline de Prétraitement
Un `ColumnTransformer` sklearn gère le prétraitement de façon reproductible :
- Variables numériques → `SimpleImputer(median)` + `StandardScaler`
- Variables catégorielles → `SimpleImputer(most_frequent)` + `OrdinalEncoder`

### Modèles Testés

| Modèle | Rôle | Justification |
|--------|------|---------------|
| Régression Logistique | Baseline | Modèle linéaire de référence, rapide et interprétable |
| Arbre de Décision | Intermédiaire | Non-linéaire, visualisable, gère bien les catégorielles |
| Random Forest | Final | Ensembliste par bagging, meilleur compromis biais-variance |

### Métriques d'Évaluation
L'accuracy seule est insuffisante sur des classes déséquilibrées. On utilise :

| Métrique | Pourquoi |
|----------|----------|
| **F1-macro** | Principale — moyenne F1 par classe sans pondération, pénalise les classes ignorées |
| **Precision-macro** | Quand le modèle prédit "Fatal", a-t-il raison ? |
| **Recall-macro** | Parmi tous les accidents fatals, combien ont été détectés ? |
| **ROC-AUC (OvR)** | Capacité de discrimination globale, indépendante du seuil |
| **Accuracy** | Fournie pour référence, mais non utilisée seule |

### Optimisation
`GridSearchCV` exhaustif sur le Random Forest — toutes les combinaisons testées, scoring sur F1-macro :

```python
param_grid = {
    'classifier__n_estimators':      [50, 100, 150],
    'classifier__max_depth':         [None, 8, 12],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__max_features':      ['sqrt', 'log2']
}
```

---

## Résultats

## Résultats de la Modélisation


| Modèle | Accuracy | F1-macro | Precision | Recall | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8357 | 0.3241 | 0.3121 | 0.3391 | 0.5822 |
| Decision Tree | 0.7212 | 0.3855 | 0.3913 | 0.3802 | 0.6146 |
| Random Forest | 0.8421 | 0.3413 | 0.3522 | 0.3311 | 0.6413 |
| **RF + GridSearchCV** ⭐ | **0.8493** | **0.4529** | **0.4486** | **0.4611** | **0.7143** |
> Valeurs à compléter après exécution complète du notebook `02_Modeling_final.ipynb`.

---

## Limites Identifiées

- `Fatal Injury` représente moins de 2% du dataset : même avec `class_weight='balanced'`, le recall sur cette classe reste faible
- Le dataset est limité à Addis Abeba sur une période donnée — la généralisation à d'autres contextes géographiques ou temporels n'est pas garantie sans retraining
- Deux variables numériques seulement (`Number_of_vehicles_involved`, `Number_of_casualties`) — le reste est catégoriel, ce qui limite les analyses de corrélation directe

## Perspectives

- Appliquer **SMOTE** pour enrichir synthétiquement la classe `Fatal Injury`
- Tester **XGBoost** ou **LightGBM**, plus efficaces sur données tabulaires déséquilibrées
- Enrichir le dataset avec des variables externes (météo précise, données GPS, vitesse)

---

## Dépendances

```
pandas >= 1.5.0
numpy >= 1.23.0
matplotlib >= 3.6.0
seaborn >= 0.12.0
scikit-learn >= 1.2.0
streamlit >= 1.20.0
```

Installation : `pip install -r requirements.txt`
