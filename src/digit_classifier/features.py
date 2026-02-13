import numpy as np
from skimage.feature import hog
from skimage.transform import resize


def extract_hog_features(
    images,
    target_size=None,
    orientations=9,
    pixels_per_cell=(4, 4),
    cells_per_block=(1, 1),
):
    """
    Extrait des caractéristiques HOG à partir d'images (2D).

    Paramètres :
    - images : liste/array d'images (ex: digits.images)
    - target_size : tuple (H, W) si on veut redimensionner (ex: (32, 32)), sinon None
    - orientations, pixels_per_cell, cells_per_block : paramètres HOG

    Retour :
    - np.array de shape (n_samples, n_features)
    """
    feats = []

    for img in images:
        img_proc = img

        if target_size is not None:
            img_proc = resize(
                img_proc,
                target_size,
                order=1,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True
            ).astype(np.float32)

        f = hog(
            img_proc,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False,
            feature_vector=True
        )
        feats.append(f)

    return np.array(feats, dtype=np.float32)
