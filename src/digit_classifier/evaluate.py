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
    Évalue un modèle entraîné et génère la matrice de confusion.

    Paramètres :
    - features_type : "pixels" ou "hog"
    - model_type : "logreg" ou "svm"
    - model_path : chemin du modèle (si None, on utilise le nom automatique)
    - output_path : chemin de l'image confusion matrix (si None, nom automatique)
    """
    if model_path is None:
        model_path = f"models/digit_model_{features_type}_{model_type}.joblib"

    if output_path is None:
        output_path = f"reports/confusion_matrix_{features_type}_{model_type}.png"

    # Données (même representation que celle utilisée à l'entraînement)
    dataset = DigitsDataset(features_type=features_type)
    _, X_test, _, y_test = dataset.get_train_test_split()

    # Chargement du modèle
    model = load_model(model_path)

    # Prédiction + score
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    # Matrice de confusion
    img_path = save_confusion_matrix(y_test, y_pred, output_path=output_path)
    return accuracy, img_path


if __name__ == "__main__":
    acc, img = evaluate_model()
    print(f"Évaluation terminée. Accuracy : {acc:.3f}")
    print(f"Matrice de confusion : {img}")
