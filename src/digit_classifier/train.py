from pathlib import Path

from src.digit_classifier.data import DigitsDataset
from src.digit_classifier.model import DigitClassifier


def train_and_save(model_path="models/digit_model.joblib"):
    """
    Entraîne le modèle de classification sur le dataset des chiffres
    et sauvegarde le modèle entraîné sur le disque.
    """
    # Chargement des données
    dataset = DigitsDataset()
    X_train, X_test, y_train, y_test = dataset.get_train_test_split()

    # Initialisation et entraînement du modèle
    classifier = DigitClassifier()
    classifier.fit(X_train, y_train)

    # Création du dossier de sauvegarde si nécessaire
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde du modèle
    classifier.save_model(model_path)

    # Évaluation rapide sur le jeu de test
    accuracy = classifier.score(X_test, y_test)
    return accuracy


if __name__ == "__main__":
    acc = train_and_save()
    print(f"Modèle entraîné et sauvegardé avec une accuracy de {acc:.3f}")
