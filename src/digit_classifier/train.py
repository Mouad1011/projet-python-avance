from pathlib import Path

from src.digit_classifier.data import DigitsDataset
from src.digit_classifier.model import DigitClassifier


def train_and_save(
    features_type="pixels",
    model_type="logreg",
    model_path=None
):
    """
    Entraîne un modèle sur le dataset digits et sauvegarde le modèle entraîné.

    Paramètres :
    - features_type : "pixels" ou "hog"
    - model_type : "logreg" ou "svm"
    - model_path : chemin de sauvegarde (optionnel). Si None, on génère un nom automatique.
    """
    # Nom automatique pour éviter d’écraser les modèles
    if model_path is None:
        model_path = f"models/digit_model_{features_type}_{model_type}.joblib"

    # Données
    dataset = DigitsDataset(features_type=features_type)
    X_train, X_test, y_train, y_test = dataset.get_train_test_split()

    # Modèle
    classifier = DigitClassifier(model_type=model_type)
    classifier.fit(X_train, y_train)

    # Sauvegarde
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(model_path)

    # Score
    accuracy = classifier.score(X_test, y_test)
    return accuracy, model_path


if __name__ == "__main__":
    acc, path = train_and_save()
    print(f"Modèle entraîné et sauvegardé dans {path} (accuracy={acc:.3f})")
