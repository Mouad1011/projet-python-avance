from skimage.feature import hog
import numpy as np


def extract_hog_features(images):
    """
    Extrait les descripteurs HOG pour un ensemble d'images.

    Paramètre :
    - images : tableau d'images 8x8 (digits.images)

    Retour :
    - tableau numpy contenant les vecteurs de features HOG
    """
    hog_features = []

    for img in images:
        features = hog(
            img,
            pixels_per_cell=(4, 4),
            cells_per_block=(1, 1),
            orientations=9,
            visualize=False,
            feature_vector=True
        )
        hog_features.append(features)

    return np.array(hog_features)
