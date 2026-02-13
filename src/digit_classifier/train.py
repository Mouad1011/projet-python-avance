from pathlib import Path

from src.digit_classifier.data import DigitsDataset
from src.digit_classifier.model import build_model, save_model


def train_and_save(
    features_type="pixels",
    model_type="logreg",
    model_path=None
):
    """
    Entraîne un modèle sur Digits et sauvegarde le modèle.

    Paramètres :
    - features_type : "pixels" ou "hog" (type de représentation)
    - model_type    : "logreg" ou "svm"
    - model_path    : chemin de sauvegarde (si None -> nom automatique)

    Retour :
    - (accuracy, model_path)
    """
    # Nom automatique : permet de garder plusieurs modèles (pixels/logreg, hog/svm, etc.)
    if model_path is None:
        model_path = f"models/digit_model_{features_type}_{model_type}.joblib"

    # Chargement et préparation des données (split train/test)
    dataset = DigitsDataset(features_type=features_type)
    X_train, X_test, y_train, y_test = dataset.get_train_test_split()

    # Construction + entraînement du modèle
    model = build_model(model_type=model_type)
    model.fit(X_train, y_train)

    # Sauvegarde du modèle entraîné
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, model_path)

    # Score simple sur le test set (accuracy)
    accuracy = model.score(X_test, y_test)
    return accuracy, model_path


if __name__ == "__main__":
    acc, path = train_and_save()
    print(f"Modèle entraîné et sauvegardé dans {path} (accuracy={acc:.3f})")
