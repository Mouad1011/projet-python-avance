from src.digit_classifier.data import DigitsDataset
from src.digit_classifier.model import load_model
from src.digit_classifier.viz import save_confusion_matrix


def evaluate_model(
    features_type="pixels",
    model_type="logreg",
    model_path=None,
    output_path=None
):
    """
    Évalue un modèle entraîné sur le test set et génère la matrice de confusion.

    - Charge les données (mêmes features que l'entraînement)
    - Charge le modèle (joblib)
    - Calcule accuracy + prédictions
    - Sauvegarde la confusion matrix dans reports/

    Retour :
    - (accuracy, img_path)
    """
    if model_path is None:
        model_path = f"models/digit_model_{features_type}_{model_type}.joblib"

    if output_path is None:
        output_path = f"reports/confusion_matrix_{features_type}_{model_type}.png"

    dataset = DigitsDataset(features_type=features_type)
    _, X_test, _, y_test = dataset.get_train_test_split()

    model = load_model(model_path)

    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    img_path = save_confusion_matrix(y_test, y_pred, output_path=output_path)
    return accuracy, img_path


if __name__ == "__main__":
    acc, img = evaluate_model()
    print(f"Évaluation terminée. Accuracy : {acc:.3f}")
    print(f"Matrice de confusion : {img}")
