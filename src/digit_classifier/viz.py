from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(y_true, y_pred, output_path="reports/confusion_matrix.png"):
    """
    Génère et sauvegarde une matrice de confusion.

    But : visualiser les erreurs (ex: 3 confondu avec 8, etc.)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path
