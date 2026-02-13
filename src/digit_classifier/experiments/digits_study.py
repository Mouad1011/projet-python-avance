from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.digit_classifier.features import extract_hog_features
from src.digit_classifier.model import build_model


def _prepare_X(digits, representation):
    """
    representation:
    - "pixels"
    - "hog8"
    - "hog32"
    """
    if representation == "pixels":
        return digits.data

    if representation == "hog8":
        return extract_hog_features(
            digits.images,
            target_size=None,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(1, 1),
        )

    if representation == "hog32":
        return extract_hog_features(
            digits.images,
            target_size=(32, 32),
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
        )

    raise ValueError("representation invalide")


def _param_grid_for(model_type):
    if model_type == "logreg":
        return {
            "clf__C": [0.1, 1, 10],
        }

    if model_type == "svm":
        return {
            "clf__kernel": ["linear", "rbf"],
            "clf__C": [0.1, 1, 10, 50, 100],
            "clf__gamma": ["scale", 0.01, 0.001],  # utile pour rbf
        }

    raise ValueError("model_type invalide")


def run_digits_study(
    out_csv="reports/digits_cv_results.csv",
    out_json="reports/digits_cv_best.json",
    cv_splits=5,
    random_state=42
):
    """
    Lance une étude comparative sur load_digits avec validation croisée.
    Sauvegarde :
    - un CSV avec toutes les configs
    - un JSON avec la meilleure config
    """
    digits = load_digits(n_class=10)
    y = digits.target

    representations = ["pixels", "hog8", "hog32"]
    models = ["logreg", "svm"]

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    results = []

    for rep in representations:
        X = _prepare_X(digits, rep)

        for model_type in models:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", build_model(model_type=model_type))
            ])

            grid = _param_grid_for(model_type)

            search = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                scoring="accuracy",
                cv=cv,
                n_jobs=-1,
                verbose=0
            )

            search.fit(X, y)

            results.append({
                "representation": rep,
                "model": model_type,
                "best_cv_accuracy": float(search.best_score_),
                "best_params": search.best_params_,
            })

            print(f"[OK] {rep} + {model_type} -> {search.best_score_:.4f}")

    df = pd.DataFrame(results).sort_values("best_cv_accuracy", ascending=False)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    best = df.iloc[0].to_dict()
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    print("\n=== TOP 5 configs ===")
    print(df.head(5).to_string(index=False))

    return df


if __name__ == "__main__":
    run_digits_study()
