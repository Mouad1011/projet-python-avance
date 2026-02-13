from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from src.digit_classifier.features import extract_hog_features


class DigitsDataset:
    """
    Charge le dataset des chiffres manuscrits et prépare les données
    soit en pixels bruts (baseline), soit en descripteurs HOG (avancé).
    """

    def __init__(self, test_size=0.2, random_state=42, features_type="pixels"):
        """
        Paramètres :
        - test_size : proportion du dataset utilisée pour le test
        - random_state : graine aléatoire pour reproductibilité
        - features_type : "pixels" ou "hog"
        """
        self.test_size = test_size
        self.random_state = random_state
        self.features_type = features_type

        self.data = None
        self.labels = None

    def load_data(self):
        digits = load_digits()
        self.labels = digits.target

        # Baseline : pixels (64 valeurs)
        if self.features_type == "pixels":
            self.data = digits.data

        # Avancé : HOG calculé depuis les images 8x8
        elif self.features_type == "hog":
            self.data = extract_hog_features(digits.images)

        else:
            raise ValueError('features_type doit être "pixels" ou "hog"')

        return self.data, self.labels

    def get_train_test_split(self):
        if self.data is None or self.labels is None:
            self.load_data()

        X_train, X_test, y_train, y_test = train_test_split(
            self.data,
            self.labels,
            test_size=self.test_size,
            random_state=self.random_state
        )

        return X_train, X_test, y_train, y_test
