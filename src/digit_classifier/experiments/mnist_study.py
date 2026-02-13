from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from skimage.feature import hog


def load_mnist(limit=2000, random_state=42):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(int)

    if limit is not None and limit < len(X):
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), size=limit, replace=False)
        X = X[idx]
        y = y[idx]

    return X, y


def extract_hog_mnist(X_flat):
    X_img = X_flat.reshape(-1, 28, 28)
    feats = []
    for img in X_img:
        f = hog(
            img,
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            visualize=False,
            feature_vector=True
        )
        feats.append(f)
    return np.array(feats, dtype=np.float32)


def evaluate_cv(X, y, model, cv):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def run_mnist_study(
    out_csv="reports/mnist_cv_results.csv",
    out_json="reports/mnist_cv_best.json",
    cv_splits=2,
    random_state=42,
    limit=2000
):
    print(f"\n[INFO] MNIST study running with limit={limit}, cv_splits={cv_splits}\n")

    X, y = load_mnist(limit=limit, random_state=random_state)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    results = []

    configs = [
        ("pixels", "logreg"),
        ("pixels", "svm_linear"),
        ("hog", "logreg"),
        ("hog", "svm_linear"),
    ]

    for rep, model_type in configs:
        if rep == "pixels":
            X_rep = X
        else:
            X_rep = extract_hog_mnist(X)

        if model_type == "logreg":
            clf = LogisticRegression(max_iter=2000, C=1.0)
        else:
            clf = LinearSVC(max_iter=3000, C=1.0)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

        mean_acc, std_acc = evaluate_cv(X_rep, y, pipe, cv)
        print(f"[OK] {rep} + {model_type} -> {mean_acc:.4f} (+/- {std_acc:.4f})")

        results.append({
            "dataset": f"MNIST(limit={limit})",
            "representation": rep,
            "model": model_type,
            "cv_mean_accuracy": mean_acc,
            "cv_std": std_acc,
            "params": {"C": 1.0}
        })

    df = pd.DataFrame(results).sort_values("cv_mean_accuracy", ascending=False)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    best = df.iloc[0].to_dict()
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    print("\n=== Résultats MNIST (triés) ===")
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    run_mnist_study(limit=2000, cv_splits=2)
