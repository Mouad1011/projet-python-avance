from src.digit_classifier.data import DigitsDataset
from src.digit_classifier.model import DigitClassifier
from src.digit_classifier.viz import save_confusion_matrix


def evaluate_model(model_path="models/digit_model.joblib"):
    """
    Évalue un modèle entraîné sur le jeu de test et
    génère la matrice de confusion associée.
    """
    # Chargement des données
    dataset = DigitsDataset()
    _, X_test, _, y_test = dataset.get_train_test_split()

    # Chargement du modèle entraîné
    classifier = DigitClassifier()
    classifier.load_model(model_path)

    # Prédictions et évaluation
    y_pred = classifier.predict(X_test)
    accuracy = classifier.score(X_test, y_test)

    # Sauvegarde de la matrice de confusion
    img_path = save_confusion_matrix(y_test, y_pred)

    return accuracy, img_path


if __name__ == "__main__":
    acc, img = evaluate_model()
    print(f"Évaluation terminée. Accuracy : {acc:.3f}")
    print(f"Matrice de confusion sauvegardée dans : {img}")
