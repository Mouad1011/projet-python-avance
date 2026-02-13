import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def build_model(model_type="logreg", max_iter=2000, random_state=42):
    """
    Construit un modèle de classification.

    - logreg : baseline rapide et robuste sur petits vecteurs de features
    - svm    : modèle plus flexible (ici kernel RBF)

    Retour : un estimateur scikit-learn prêt à être entraîné (fit).
    """
    if model_type == "logreg":
        return LogisticRegression(
            max_iter=max_iter,
            random_state=random_state
        )

    if model_type == "svm":
        return SVC(kernel="rbf", gamma="scale")

    raise ValueError('model_type doit être "logreg" ou "svm"')


def save_model(model, path):
    """Sauvegarde un modèle entraîné au format joblib."""
    joblib.dump(model, path)


def load_model(path):
    """Charge un modèle sauvegardé au format joblib."""
    return joblib.load(path)
