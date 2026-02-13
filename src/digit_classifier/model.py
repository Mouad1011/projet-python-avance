import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def build_model(model_type="logreg", max_iter=2000, random_state=42):
    """
    Construit un modèle de classification.
    - logreg : baseline simple
    - svm : modèle plus flexible (kernel possible)
    """
    if model_type == "logreg":
        return LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            multi_class="auto"
        )

    if model_type == "svm":
        return SVC(kernel="rbf", gamma="scale")

    raise ValueError('model_type doit être "logreg" ou "svm"')


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
