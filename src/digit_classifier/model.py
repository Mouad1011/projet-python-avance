import joblib
from sklearn.linear_model import LogisticRegression


class DigitClassifier:
    """
    Classe regroupant le modèle de classification des chiffres manuscrits.
    Le modèle utilisé est une régression logistique de scikit-learn.
    """

    def __init__(self, max_iter=2000, random_state=42):
        """
        Paramètres :
        - max_iter : nombre maximum d'itérations pour l'entraînement
        - random_state : graine aléatoire pour la reproductibilité
        """
        self.max_iter = max_iter
        self.random_state = random_state

        self.model = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            multi_class="auto"
        )

    def fit(self, X_train, y_train):
        """
        Entraîne le modèle sur les données d'entraînement.
        """
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """
        Prédit les labels associés aux données X.
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Retourne l'accuracy du modèle sur un jeu de données donné.
        """
        return self.model.score(X, y)

    def save_model(self, path):
        """
        Sauvegarde le modèle entraîné sur le disque.
        """
        joblib.dump(self.model, path)

    def load_model(self, path):
        """
        Charge un modèle précédemment sauvegardé.
        """
        self.model = joblib.load(path)
        return self
