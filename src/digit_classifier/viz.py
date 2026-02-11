from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(y_true, y_pred, output_path="reports/confusion_matrix.png"):
    """
    Génère et sauvegarde la matrice de confusion à partir des
    labels réels et des prédictions du modèle.
    """
    # Création du dossier de sortie si nécessaire
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Génération de la matrice de confusion
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path
