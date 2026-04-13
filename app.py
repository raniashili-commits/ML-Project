import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RTA Severity Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

RANDOM_STATE = 42
TARGET       = 'Accident_severity'

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Fond principal */
    .main { background-color: #0D1B2A; }
    .stApp { background-color: #0D1B2A; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1B2A3B;
        border-right: 2px solid #2E86C1;
    }

    /* Titres */
    h1, h2, h3 { color: #FFFFFF !important; }
    p, li, label { color: #D6EAF8 !important; }

    /* Cartes metriques */
    .metric-card {
        background: linear-gradient(135deg, #1B4F72, #2E86C1);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin: 6px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #AED6F1;
        margin-top: 4px;
    }

    /* Badge gravite */
    .badge-slight   { background:#1E8449; color:#fff; padding:6px 14px; border-radius:20px; font-weight:700; }
    .badge-serious  { background:#D35400; color:#fff; padding:6px 14px; border-radius:20px; font-weight:700; }
    .badge-fatal    { background:#C0392B; color:#fff; padding:6px 14px; border-radius:20px; font-weight:700; }

    /* Boutons */
    .stButton > button {
        background: linear-gradient(135deg, #1B4F72, #2E86C1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2E86C1, #1B4F72);
        transform: translateY(-1px);
    }

    /* Inputs */
    .stSelectbox > div > div, .stNumberInput > div > div {
        background-color: #1B2A3B !important;
        color: white !important;
        border: 1px solid #2E86C1 !important;
        border-radius: 6px !important;
    }

    /* Séparateur */
    hr { border-color: #2E86C1 !important; opacity: 0.4; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background-color: #1B2A3B;
        color: #AED6F1;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E86C1 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file):
    """Charge, shuffle et echantillonne le dataset."""
    df_full = pd.read_csv(uploaded_file)
    df_full = df_full.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df      = df_full.head(3000).copy()
    return df_full, df


def build_pipeline(preprocessor, model):
    return Pipeline([('preprocessor', preprocessor), ('classifier', model)])


@st.cache_resource
def train_models(df):
    """Entraine les 3 modeles et retourne les resultats."""
    le = LabelEncoder()
    y  = le.fit_transform(df[TARGET])
    X  = df.drop(columns=[TARGET])

    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    # Feature engineering
    X = X.copy()
    if 'Time' in X.columns:
        X['Hour'] = pd.to_datetime(X['Time'], format='%H:%M:%S', errors='coerce').dt.hour
        X['Time_Period'] = pd.cut(
            X['Hour'].fillna(0), bins=[0,6,12,18,24],
            labels=['Nuit','Matin','Apres-midi','Soir'], right=False
        ).astype(str)
        cat_cols.append('Time_Period')
    if 'Day_of_week' in X.columns and 'Light_conditions' in X.columns:
        X['Day_Light'] = X['Day_of_week'].astype(str) + '_' + X['Light_conditions'].astype(str)
        cat_cols.append('Day_Light')

    # Pipeline
    num_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models_def = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        'Decision Tree':       DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=RANDOM_STATE),
        'Random Forest':       RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    }

    results = {}
    for name, clf in models_def.items():
        pipe = build_pipeline(preprocessor, clf)
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        try:
            y_proba = pipe.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        except:
            auc = np.nan
        results[name] = {
            'pipe': pipe,
            'cv_f1': cv_f1.mean(),
            'accuracy':  accuracy_score(y_test, y_pred),
            'f1':        f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall':    recall_score(y_test, y_pred, average='macro', zero_division=0),
            'auc':       auc,
            'y_pred':    y_pred
        }

    # GridSearchCV sur Random Forest
    param_grid = {
        'classifier__n_estimators':      [50, 100, 150],
        'classifier__max_depth':         [None, 8, 12],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__max_features':      ['sqrt', 'log2']
    }
    best_pipe = build_pipeline(
        preprocessor,
        RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    )
    gs = GridSearchCV(best_pipe, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    y_opt = gs.best_estimator_.predict(X_test)
    try:
        y_opt_proba = gs.best_estimator_.predict_proba(X_test)
        auc_opt = roc_auc_score(y_test, y_opt_proba, multi_class='ovr', average='macro')
    except:
        auc_opt = np.nan

    results['RF + GridSearch ⭐'] = {
        'pipe': gs.best_estimator_,
        'cv_f1': gs.best_score_,
        'accuracy':  accuracy_score(y_test, y_opt),
        'f1':        f1_score(y_test, y_opt, average='macro'),
        'precision': precision_score(y_test, y_opt, average='macro', zero_division=0),
        'recall':    recall_score(y_test, y_opt, average='macro', zero_division=0),
        'auc':       auc_opt,
        'y_pred':    y_opt,
        'best_params': gs.best_params_
    }

    return results, le, X_test, y_test, num_cols, cat_cols, preprocessor, X


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 RTA Predictor")
    st.markdown("**Prédiction de la Gravité des Accidents**")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Charger le dataset RTA (.csv)",
        type=["csv"],
        help="Charger le fichier RTA_Dataset.csv"
    )

    if uploaded_file:
        st.success("✅ Dataset chargé")

    st.markdown("---")
    st.markdown("### 📋 Infos Projet")
    st.info(
        "**Module :** Machine Learning\n\n"
        "**Encadrant :** M. Abdallah Khemais\n\n"
        "**Tâche :** Classification multiclasse\n\n"
        "**Dataset :** RTA (3 000 lignes)"
    )
    st.markdown("---")
    st.markdown("### 🎯 Métriques utilisées")
    st.markdown("""
    - **F1-macro** ← principale
    - Precision-macro
    - Recall-macro
    - ROC-AUC (OvR)
    - Accuracy (comparaison)
    """)
    st.caption("⚠️ L'accuracy seule est insuffisante sur un dataset déséquilibré.")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #0D1B2A, #1B4F72);
            padding: 32px; border-radius: 16px;
            border-left: 6px solid #2E86C1; margin-bottom: 24px;'>
    <h1 style='color:#FFFFFF; margin:0; font-size:2.2rem;'>
        🚗 Prédiction de la Gravité des Accidents de la Route
    </h1>
    <p style='color:#AED6F1; margin:8px 0 0 0; font-size:1.05rem;'>
        Projet Machine Learning End-to-End — RTA Dataset |
        Shuffle 12 300 lignes → Échantillon 3 000 | GridSearchCV | 5 Métriques
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SANS DATASET
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div style='background:#1B2A3B; border-radius:12px; padding:40px; text-align:center;
                border: 2px dashed #2E86C1; margin-top:20px;'>
        <h2 style='color:#2E86C1;'>📂 Chargez votre dataset pour commencer</h2>
        <p style='color:#AED6F1;'>
            Utilisez le panneau latéral pour charger le fichier <strong>RTA_Dataset.csv</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# CHARGEMENT & ENTRAINEMENT
# ─────────────────────────────────────────────
with st.spinner("🔄 Shuffle → Échantillon → Entraînement → GridSearchCV..."):
    df_full, df = load_data(uploaded_file)
    results, le_target, X_test, y_test, num_cols, cat_cols, preprocessor, X_feat = train_models(df)

st.success(f"✅ Dataset : {df_full.shape[0]} lignes shufflées → {df.shape[0]} lignes retenues | {len(results)} modèles entraînés")

# ─────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA", "🤖 Modèles", "🏆 Meilleur Modèle", "🔮 Prédiction", "📈 Comparaison"
])


# ══════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════
with tab1:
    st.markdown("## 📊 Analyse Exploratoire des Données")

    # Stats rapides
    col1, col2, col3, col4 = st.columns(4)
    for col, (label, val) in zip(
        [col1, col2, col3, col4],
        [
            ("Lignes (total)", f"{df_full.shape[0]:,}"),
            ("Échantillon", f"{df.shape[0]:,}"),
            ("Features", str(df.shape[1] - 1)),
            ("Classes", str(df[TARGET].nunique())),
        ]
    ):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='value'>{val}</div>
            <div class='label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Distribution de la Variable Cible")
        counts = df[TARGET].value_counts()
        pcts   = counts / len(df) * 100

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#1B2A3B')
        ax.set_facecolor('#1B2A3B')
        bars = ax.bar(counts.index, counts.values,
                      color=['#2ECC71','#E67E22','#E74C3C'], edgecolor='white', linewidth=0.5)
        for bar, pct in zip(bars, pcts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{pct:.1f}%', ha='center', fontsize=10, color='white', fontweight='bold')
        ax.set_title('Effectif par classe', color='white', fontweight='bold')
        ax.tick_params(colors='white', labelsize=8)
        ax.spines['bottom'].set_color('#2E86C1')
        ax.spines['left'].set_color('#2E86C1')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('Nombre', color='white')
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        ratio = counts.max() / counts.min()
        st.warning(
            f"⚠️ **Déséquilibre détecté** — ratio max/min = **{ratio:.0f}x**\n\n"
            f"Un modèle naïf (toujours 'Slight') = **{pcts.max():.1f}% d'accuracy** sans rien apprendre.\n\n"
            f"→ On utilise F1-macro comme métrique principale."
        )

    with col_right:
        st.markdown("### Valeurs Manquantes")
        missing = pd.DataFrame({
            'Count': df.isnull().sum(),
            'Pct (%)': (df.isnull().sum() / len(df) * 100).round(2)
        }).sort_values('Pct (%)', ascending=False)
        missing_top = missing[missing['Count'] > 0]

        if len(missing_top) == 0:
            st.success("✅ Aucune valeur manquante détectée.")
        else:
            fig, ax = plt.subplots(figsize=(6, max(3, len(missing_top) * 0.4)))
            fig.patch.set_facecolor('#1B2A3B')
            ax.set_facecolor('#1B2A3B')
            ax.barh(missing_top.index, missing_top['Pct (%)'],
                    color='#E74C3C', edgecolor='white', linewidth=0.5)
            ax.set_title('Taux de valeurs manquantes (%)', color='white', fontweight='bold')
            ax.tick_params(colors='white', labelsize=7)
            ax.spines['bottom'].set_color('#2E86C1')
            ax.spines['left'].set_color('#2E86C1')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.info("💡 **Stratégie :** imputation dans le pipeline (médiane / mode). Aucune ligne supprimée.")

    st.markdown("---")
    st.markdown("### Analyse Bivariée")
    cat_feats = [c for c in df.select_dtypes('object').columns if c != TARGET]
    feat_sel  = st.selectbox("Choisir une variable :", cat_feats[:15])

    if feat_sel:
        ct = pd.crosstab(df[feat_sel], df[TARGET], normalize='index') * 100
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#1B2A3B')
        ax.set_facecolor('#1B2A3B')
        ct.head(8).plot(kind='bar', ax=ax,
                        color=['#2ECC71','#E67E22','#E74C3C'], edgecolor='white')
        ax.set_title(f'{feat_sel} vs {TARGET}', color='white', fontweight='bold')
        ax.set_ylabel('Proportion (%)', color='white')
        ax.tick_params(colors='white', labelsize=8)
        ax.spines['bottom'].set_color('#2E86C1')
        ax.spines['left'].set_color('#2E86C1')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        legend = ax.legend(title='Gravité', facecolor='#1B2A3B', labelcolor='white')
        legend.get_title().set_color('white')
        plt.xticks(rotation=25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════
# TAB 2 — MODÈLES
# ══════════════════════════════════════════════
with tab2:
    st.markdown("## 🤖 Comparaison des Modèles")
    st.info("5 métriques calculées par modèle. **F1-macro** est la métrique principale car le dataset est déséquilibré.")

    # Tableau
    comp_data = []
    for name, res in results.items():
        comp_data.append({
            'Modèle':       name,
            'CV F1-macro':  f"{res['cv_f1']:.4f}",
            'Accuracy':     f"{res['accuracy']:.4f}",
            'F1-macro':     f"{res['f1']:.4f}",
            'Precision':    f"{res['precision']:.4f}",
            'Recall':       f"{res['recall']:.4f}",
            'ROC-AUC':      f"{res['auc']:.4f}",
        })
    st.dataframe(
        pd.DataFrame(comp_data).set_index('Modèle'),
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("### Comparaison Visuelle — F1-macro")

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#1B2A3B')
    ax.set_facecolor('#1B2A3B')
    names  = list(results.keys())
    f1vals = [results[n]['f1'] for n in names]
    colors = ['#5D6D7E','#5D6D7E','#5D6D7E','#F1C40F']
    ax.bar(names, f1vals, color=colors, edgecolor='white', linewidth=0.5)
    for i, v in enumerate(f1vals):
        ax.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=10,
                color='white', fontweight='bold')
    ax.axhline(y=df[TARGET].value_counts(normalize=True).max(),
               color='red', linestyle='--', alpha=0.6,
               label=f'Accuracy naive (~{df[TARGET].value_counts(normalize=True).max()*100:.0f}%)')
    ax.set_ylabel('F1-macro', color='white')
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors='white', labelsize=8)
    ax.spines['bottom'].set_color('#2E86C1')
    ax.spines['left'].set_color('#2E86C1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#1B2A3B', labelcolor='white')
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Matrices de Confusion")
    cols_cm = st.columns(len(results))
    for col, (name, res) in zip(cols_cm, results.items()):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.patch.set_facecolor('#1B2A3B')
            ax.set_facecolor('#1B2A3B')
            cm = confusion_matrix(y_test, res['y_pred'])
            ConfusionMatrixDisplay(cm, display_labels=le_target.classes_).plot(
                ax=ax, cmap='Blues', colorbar=False
            )
            ax.set_title(f'{name}\nF1={res["f1"]:.3f}', color='white',
                         fontweight='bold', fontsize=8)
            ax.tick_params(colors='white', labelsize=6)
            plt.xticks(rotation=25, fontsize=6)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════
# TAB 3 — MEILLEUR MODÈLE
# ══════════════════════════════════════════════
with tab3:
    st.markdown("## 🏆 Random Forest + GridSearchCV")
    best_res = results['RF + GridSearch ⭐']

    # Métriques
    metrics_labels = ['Accuracy','F1-macro','Precision','Recall','ROC-AUC']
    metrics_vals   = [
        best_res['accuracy'], best_res['f1'],
        best_res['precision'], best_res['recall'], best_res['auc']
    ]
    cols_m = st.columns(5)
    for col, label, val in zip(cols_m, metrics_labels, metrics_vals):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='value'>{val:.4f}</div>
            <div class='label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Meilleurs Hyperparamètres (GridSearchCV)")
        if 'best_params' in best_res:
            params_df = pd.DataFrame([
                {'Paramètre': k.replace('classifier__',''), 'Valeur': str(v)}
                for k, v in best_res['best_params'].items()
            ])
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        st.markdown("### Rapport de Classification")
        report = classification_report(
            y_test, best_res['y_pred'],
            target_names=le_target.classes_, output_dict=True
        )
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    with col_r:
        st.markdown("### Importance des Features (Top 15)")
        try:
            rf_clf      = best_res['pipe'].named_steps['classifier']
            importances = rf_clf.feature_importances_
            n           = len(importances)
            feat_names  = (num_cols + cat_cols)[:n]
            imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})\
                       .sort_values('Importance', ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(6, 6))
            fig.patch.set_facecolor('#1B2A3B')
            ax.set_facecolor('#1B2A3B')
            ax.barh(imp_df['Feature'][::-1], imp_df['Importance'][::-1],
                    color='#2E86C1', edgecolor='white', linewidth=0.3)
            ax.set_title('Feature Importance', color='white', fontweight='bold')
            ax.tick_params(colors='white', labelsize=7)
            ax.spines['bottom'].set_color('#2E86C1')
            ax.spines['left'].set_color('#2E86C1')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"Feature importance non disponible : {e}")


# ══════════════════════════════════════════════
# TAB 4 — PRÉDICTION
# ══════════════════════════════════════════════
with tab4:
    st.markdown("## 🔮 Prédiction en Temps Réel")
    st.markdown("Renseignez les caractéristiques de l'accident pour prédire sa gravité.")

    best_pipe = results['RF + GridSearch ⭐']['pipe']

    # On prend les colonnes originales du dataset (avant feature eng)
    original_cols = df.drop(columns=[TARGET]).columns.tolist()
    sample_row    = df.drop(columns=[TARGET]).iloc[0].to_dict()

    cat_input_cols = df.drop(columns=[TARGET]).select_dtypes('object').columns.tolist()
    num_input_cols = df.drop(columns=[TARGET]).select_dtypes(['int64','float64']).columns.tolist()

    input_data = {}

    with st.form("prediction_form"):
        st.markdown("### Variables Catégorielles")
        n_cat = len(cat_input_cols)
        cat_col_groups = [cat_input_cols[i:i+3] for i in range(0, n_cat, 3)]
        for group in cat_col_groups:
            cols_form = st.columns(len(group))
            for c, feat in zip(cols_form, group):
                options = df[feat].dropna().unique().tolist()
                input_data[feat] = c.selectbox(
                    feat, options,
                    index=0,
                    key=f"cat_{feat}"
                )

        if num_input_cols:
            st.markdown("### Variables Numériques")
            num_col_groups = [num_input_cols[i:i+3] for i in range(0, len(num_input_cols), 3)]
            for group in num_col_groups:
                cols_num = st.columns(len(group))
                for c, feat in zip(cols_num, group):
                    mn = float(df[feat].min())
                    mx = float(df[feat].max())
                    dv = float(df[feat].median())
                    input_data[feat] = c.number_input(
                        feat, min_value=mn, max_value=mx, value=dv, key=f"num_{feat}"
                    )

        submitted = st.form_submit_button("🔮 Prédire la Gravité")

    if submitted:
        input_df = pd.DataFrame([input_data])

        # Feature engineering sur l'input
        if 'Time' in input_df.columns:
            input_df['Hour'] = pd.to_datetime(
                input_df['Time'], format='%H:%M:%S', errors='coerce'
            ).dt.hour
            input_df['Time_Period'] = pd.cut(
                input_df['Hour'].fillna(0),
                bins=[0,6,12,18,24],
                labels=['Nuit','Matin','Apres-midi','Soir'], right=False
            ).astype(str)
        if 'Day_of_week' in input_df.columns and 'Light_conditions' in input_df.columns:
            input_df['Day_Light'] = (
                input_df['Day_of_week'].astype(str) + '_' +
                input_df['Light_conditions'].astype(str)
            )

        try:
            pred       = best_pipe.predict(input_df)[0]
            pred_label = le_target.inverse_transform([pred])[0]
            proba      = best_pipe.predict_proba(input_df)[0]

            badge_class = {
                'Slight Injury':  'badge-slight',
                'Serious Injury': 'badge-serious',
                'Fatal Injury':   'badge-fatal'
            }.get(pred_label, 'badge-slight')

            st.markdown("---")
            st.markdown("### Résultat")

            col_pred, col_prob = st.columns(2)
            with col_pred:
                st.markdown(f"""
                <div style='background:#1B2A3B; border-radius:12px; padding:24px;
                            border:2px solid #2E86C1; text-align:center;'>
                    <p style='color:#AED6F1; font-size:0.9rem; margin:0;'>Gravité prédite</p>
                    <span class='{badge_class}' style='font-size:1.4rem; display:inline-block; margin:12px 0;'>
                        {pred_label}
                    </span>
                    <p style='color:#AED6F1; font-size:0.85rem;'>
                        Confiance : <strong style='color:white;'>{proba[pred]*100:.1f}%</strong>
                    </p>
                </div>""", unsafe_allow_html=True)

            with col_prob:
                fig, ax = plt.subplots(figsize=(5, 3))
                fig.patch.set_facecolor('#1B2A3B')
                ax.set_facecolor('#1B2A3B')
                colors_p = ['#2ECC71','#E67E22','#E74C3C']
                ax.barh(le_target.classes_, proba, color=colors_p, edgecolor='white')
                ax.set_xlim(0, 1)
                ax.set_title('Probabilités par classe', color='white', fontweight='bold', fontsize=10)
                ax.tick_params(colors='white', labelsize=8)
                ax.spines['bottom'].set_color('#2E86C1')
                ax.spines['left'].set_color('#2E86C1')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                for i, p in enumerate(proba):
                    ax.text(p + 0.01, i, f'{p*100:.1f}%', va='center',
                            color='white', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        except Exception as e:
            st.error(f"Erreur de prédiction : {e}")


# ══════════════════════════════════════════════
# TAB 5 — COMPARAISON GLOBALE
# ══════════════════════════════════════════════
with tab5:
    st.markdown("## 📈 Synthèse Complète")

    st.markdown("""
    ### Pourquoi plusieurs métriques ?

    | Métrique | Définition | Pertinence |\n    |----------|-----------|------------|\n    | **Accuracy** | (TP+TN)/Total | Insuffisante seule sur données déséquilibrées |\n    | **F1-macro** | Moyenne F1 par classe (sans pondération) | **Principale** — pénalise classes ignorées |\n    | **Precision** | TP/(TP+FP) | Quand je prédit Fatal, ai-je raison ? |\n    | **Recall** | TP/(TP+FN) | Parmi tous les fatals, combien détectés ? |\n    | **ROC-AUC** | Discrimination globale | Vue d'ensemble indépendante du seuil |
    """)

    st.markdown("---")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1B2A3B')

    # Radar des métriques
    metric_keys = ['accuracy','f1','precision','recall','auc']
    metric_labels = ['Accuracy','F1-macro','Precision','Recall','AUC']
    model_names = list(results.keys())
    model_colors = ['#3498DB','#2ECC71','#E67E22','#F1C40F']

    ax = axes[0]
    ax.set_facecolor('#1B2A3B')
    x = np.arange(len(metric_labels))
    width = 0.2
    for i, (name, color) in enumerate(zip(model_names, model_colors)):
        vals = [results[name][k] for k in metric_keys]
        ax.bar(x + i*width, vals, width, label=name, color=color, edgecolor='white', linewidth=0.3)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_labels, rotation=15, color='white', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title('Toutes les métriques — Tous les modèles', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#2E86C1')
    ax.spines['left'].set_color('#2E86C1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leg = ax.legend(facecolor='#0D1B2A', labelcolor='white', fontsize=7)

    # Gain GridSearchCV
    ax2 = axes[1]
    ax2.set_facecolor('#1B2A3B')
    rf_vals   = [results['Random Forest'][k] for k in metric_keys]
    best_vals = [results['RF + GridSearch ⭐'][k] for k in metric_keys]
    gains     = [b - r for b, r in zip(best_vals, rf_vals)]
    colors_g  = ['#2ECC71' if g >= 0 else '#E74C3C' for g in gains]
    ax2.bar(metric_labels, gains, color=colors_g, edgecolor='white', linewidth=0.3)
    for i, g in enumerate(gains):
        ax2.text(i, g + 0.001 if g >= 0 else g - 0.003,
                 f'{g:+.4f}', ha='center', fontsize=9, color='white', fontweight='bold')
    ax2.axhline(0, color='white', linewidth=0.5)
    ax2.set_title('Gain apporté par GridSearchCV\n(RF Optimisé − RF de base)',
                  color='white', fontweight='bold')
    ax2.tick_params(colors='white', labelsize=9)
    ax2.spines['bottom'].set_color('#2E86C1')
    ax2.spines['left'].set_color('#2E86C1')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("### Conclusion")
    best_f1_name = max(results, key=lambda k: results[k]['f1'])
    best_f1_val  = results[best_f1_name]['f1']
    st.success(
        f"🏆 **Meilleur modèle : {best_f1_name}** — "
        f"F1-macro = **{best_f1_val:.4f}** | "
        f"AUC = **{results[best_f1_name]['auc']:.4f}**\n\n"
        f"GridSearchCV a permis une optimisation exhaustive de la grille "
        f"({3*3*3*2} combinaisons × 5 folds) avec F1-macro comme critère."
    )
