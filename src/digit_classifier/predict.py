import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from src.digit_classifier.model import DigitClassifier


def predict_by_index(index, model_path="models/digit_model.joblib"):
    """
    Prédit le chiffre correspondant à une image du dataset
    à partir de son index et affiche l'image associée.
    """
    # Chargement du dataset
    digits = load_digits()
    data = digits.data
    labels = digits.target
    images = digits.images

    # Chargement du modèle entraîné
    classifier = DigitClassifier()
    classifier.load_model(model_path)

    # Prédiction pour l'index donné
    sample = data[index].reshape(1, -1)
    prediction = int(classifier.predict(sample)[0])
    true_label = int(labels[index])

    # Affichage de l'image et du résultat
    plt.imshow(images[index], cmap="gray")
    plt.title(f"Prédiction : {prediction} | Vrai label : {true_label}")
    plt.axis("off")
    plt.show()

    return prediction, true_label


if __name__ == "__main__":
    idx = 0
    pred, true = predict_by_index(idx)
    print(f"Index {idx} -> prédiction = {pred}, label réel = {true}")
