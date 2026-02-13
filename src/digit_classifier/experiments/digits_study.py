from pathlib import Path
import json

import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.digit_classifier.features import extract_hog_features
from src.digit_classifier.model import build_model


def _prepare_X(digits, representation):
    """
    Prépare la matrice de features X selon la représentation choisie.

    Représentations :
    - "pixels" : baseline (pixels bruts, déjà fournis par sklearn sous forme de vecteurs)
    - "hog8"   : HOG directement sur les images 8x8 (peut être limité car images petites)
    - "hog32"  : redimensionnement en 32x32 puis HOG (plus d'information pour les gradients)

    Retour :
    - X : np.array de shape (n_samples, n_features)
    """
    if representation == "pixels":
        # digits.data est déjà de forme (n_samples, 64) pour des images 8x8
        return digits.data

    if representation == "hog8":
        # HOG sans redimensionnement (sur 8x8)
        return extract_hog_features(
            digits.images,
            target_size=None,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(1, 1),
        )

    if representation == "hog32":
        # HOG après redimensionnement 32x32 :
        # objectif : rendre les contours/gradients plus exploitables
        return extract_hog_features(
            digits.images,
            target_size=(32, 32),
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
        )

    raise ValueError("representation invalide (attendu: pixels, hog8, hog32)")


def _param_grid_for(model_type):
    """
    Définit une grille d'hyperparamètres différente selon le modèle.

    - logreg : on ajuste C (force de régularisation)
    - svm    : on ajuste kernel + C + gamma (gamma utile surtout pour RBF)
    """
    if model_type == "logreg":
        return {
            "clf__C": [0.1, 1, 10],
        }

    if model_type == "svm":
        return {
            "clf__kernel": ["linear", "rbf"],
            "clf__C": [0.1, 1, 10, 50, 100],
            "clf__gamma": ["scale", 0.01, 0.001],
        }

    raise ValueError("model_type invalide (attendu: logreg ou svm)")


def run_digits_study(
    out_csv="reports/digits_cv_results.csv",
    out_json="reports/digits_cv_best.json",
    cv_splits=5,
    random_state=42
):
    """
    Lance une étude comparative sur load_digits avec validation croisée.

    Objectif :
    - comparer plusieurs représentations (pixels vs HOG) et plusieurs modèles (logreg vs SVM)
    - optimiser les hyperparamètres via GridSearchCV
    - sauvegarder les résultats pour le compte rendu

    Fichiers générés :
    - CSV : toutes les configurations testées
    - JSON : meilleure configuration (top-1)
    """
    # Dataset Digits : images 8x8 + labels (0..9)
    digits = load_digits(n_class=10)
    y = digits.target

    # Représentations testées
    representations = ["pixels", "hog8", "hog32"]

    # Modèles testés
    models = ["logreg", "svm"]

    # Validation croisée stratifiée : conserve la proportion de classes à chaque fold
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    results = []

    # Boucle principale : comparaison systématique de toutes les combinaisons
    for rep in representations:
        # Préparer les features X selon la représentation
        X = _prepare_X(digits, rep)

        for model_type in models:
            # Pipeline scikit-learn :
            # - StandardScaler : important pour logreg et surtout SVM (sensibles à l'échelle)
            # - clf : modèle (logreg ou svm)
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", build_model(model_type=model_type))
            ])

            # Grille d'hyperparamètres spécifique au modèle
            grid = _param_grid_for(model_type)

            # GridSearchCV :
            # - évalue chaque combinaison (rep, modèle, hyperparamètres) en CV
            # - garde le meilleur modèle selon l'accuracy moyenne CV
            search = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                scoring="accuracy",
                cv=cv,
                n_jobs=-1,     # utilise tous les coeurs disponibles
                verbose=0
            )

            # Entraînement + validation croisée sur l'ensemble du dataset
            search.fit(X, y)

            # Stockage des résultats principaux
            results.append({
                "representation": rep,
                "model": model_type,
                "best_cv_accuracy": float(search.best_score_),
                "best_params": search.best_params_,
            })

            print(f"[OK] {rep} + {model_type} -> {search.best_score_:.4f}")

    # Tableau des résultats trié par performance décroissante
    df = pd.DataFrame(results).sort_values("best_cv_accuracy", ascending=False)

    # Sauvegarde CSV (toutes les configurations)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Sauvegarde JSON (meilleure configuration)
    best = df.iloc[0].to_dict()
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    # Affichage console (utile pour debug / vérification rapide)
    print("\n=== TOP 5 configs ===")
    print(df.head(5).to_string(index=False))

    return df


if __name__ == "__main__":
    run_digits_study()
