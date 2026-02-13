import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from src.digit_classifier.features import extract_features
from src.digit_classifier.model import load_model


def predict_by_index(
    index,
    features_type="pixels",
    model_type="logreg",
    model_path=None,
    hog_target_size=None
):
    """
    Prédit le chiffre correspondant à l'image 'index' du dataset Digits.

    - Charge l'image (pour l'affichage) et le label réel
    - Construit les features (pixels ou HOG)
    - Charge le modèle entraîné
    - Prédit + affiche l'image avec (prediction / vrai label)

    Paramètres :
    - index : index de l'image dans Digits
    - features_type : "pixels" ou "hog"
    - model_type : "logreg" ou "svm"
    - model_path : chemin du modèle (si None -> nom auto)
    - hog_target_size : ex (32,32) si le modèle a été entraîné sur HOG après resize

    Retour :
    - (prediction, true_label)
    """
    if model_path is None:
        model_path = f"models/digit_model_{features_type}_{model_type}.joblib"

    digits = load_digits()
    image = digits.images[index]
    true_label = int(digits.target[index])

    # On met l'image dans un batch (n=1) pour respecter la forme attendue par extract_features
    X_img = np.array([image], dtype=np.float32)

    # Features (pixels ou HOG) : doit correspondre à l'entraînement
    X_feat = extract_features(
        X_img,
        features_type=features_type,
        target_size=hog_target_size
    )

    model = load_model(model_path)
    prediction = int(model.predict(X_feat)[0])

    plt.imshow(image, cmap="gray")
    plt.title(f"Prédiction : {prediction} | Vrai label : {true_label}")
    plt.axis("off")
    plt.show()

    return prediction, true_label


if __name__ == "__main__":
    pred, true = predict_by_index(42, features_type="pixels", model_type="logreg")
    print(f"Index 42 -> prédiction = {pred}, label réel = {true}")
