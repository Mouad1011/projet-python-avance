from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class DigitsDataset:
    """
    Classe utilitaire pour charger le dataset des chiffres manuscrits
    et le découper en ensembles d'entraînement et de test.
    """

    def __init__(self, test_size=0.2, random_state=42):
        """
        Paramètres :
        - test_size : proportion des données utilisée pour le test
        - random_state : graine aléatoire pour garantir la reproductibilité
        """
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.labels = None

    def load_data(self):
        """
        Charge le dataset load_digits de scikit-learn.
        Les images (8x8) sont automatiquement aplaties en vecteurs de pixels.
        """
        digits = load_digits()

        self.data = digits.data      # vecteurs de pixels (64 valeurs)
        self.labels = digits.target  # labels réels (0 à 9)

        return self.data, self.labels

    def get_train_test_split(self):
        """
        Sépare les données en ensembles d'entraînement et de test.
        """
        if self.data is None or self.labels is None:
            self.load_data()

        X_train, X_test, y_train, y_test = train_test_split(
            self.data,
            self.labels,
            test_size=self.test_size,
            random_state=self.random_state
        )

        return X_train, X_test, y_train, y_test
