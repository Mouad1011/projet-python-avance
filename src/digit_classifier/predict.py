import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from src.digit_classifier.features import extract_features
from src.digit_classifier.model import load_model


def predict_by_index(
    index,
    features_type="pixels",
    model_type="logreg",
    model_path=None
):
    """
    Prédit le chiffre correspondant à une image du dataset digits
    à partir de son index et affiche l'image associée.

    Paramètres :
    - index : index de l'image dans le dataset
    - features_type : "pixels" ou "hog"
    - model_type : "logreg" ou "svm"
    - model_path : chemin du modèle (si None, nom automatique)
    """
    if model_path is None:
        model_path = f"models/digit_model_{features_type}_{model_type}.joblib"

    digits = load_digits()
    image = digits.images[index]          # (8,8)
    true_label = int(digits.target[index])

    # Préparer l'entrée sous forme attendue par extract_features()
    # -> shape (n_samples, 8, 8)
    X_img = np.array([image], dtype=np.float32)

    # Extraction des features selon features_type
    X_feat = extract_features(X_img, features_type=features_type)  # (1, n_features)

    # Chargement du modèle et prédiction
    model = load_model(model_path)
    prediction = int(model.predict(X_feat)[0])

    # Affichage
    plt.imshow(image, cmap="gray")
    plt.title(f"Prédiction : {prediction} | Vrai label : {true_label}")
    plt.axis("off")
    plt.show()

    return prediction, true_label


if __name__ == "__main__":
    idx = 0
    pred, true = predict_by_index(idx, features_type="pixels", model_type="logreg")
    print(f"Index {idx} -> prédiction = {pred}, label réel = {true}")
