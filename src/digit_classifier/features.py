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

    Idée :
    - HOG (Histogram of Oriented Gradients) décrit les contours via les gradients.
    - C'est souvent plus pertinent que les pixels bruts pour des chiffres manuscrits.

    Paramètres :
    - images : array/list d'images (n_samples, H, W)
    - target_size : (H, W) pour redimensionner avant HOG (ex : (32, 32)), sinon None
    - orientations, pixels_per_cell, cells_per_block : hyperparamètres HOG

    Retour :
    - np.array (n_samples, n_features) : vecteurs HOG pour chaque image
    """
    feats = []

    for img in images:
        img_proc = img

        # Option : redimensionnement pour que HOG ait plus d'information (utile si images trop petites)
        if target_size is not None:
            img_proc = resize(
                img_proc,
                target_size,
                order=1,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True
            ).astype(np.float32)

        # Extraction HOG : on récupère un vecteur de features (feature_vector=True)
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


def extract_features(
    images,
    features_type="pixels",
    target_size=None,
    orientations=9,
    pixels_per_cell=(4, 4),
    cells_per_block=(1, 1),
):
    """
    Fonction "wrapper" pour extraire des features selon le choix.

    - pixels : aplatissement (H*W) -> vecteur
    - hog    : extraction HOG (avec redimensionnement optionnel)

    Objectif : avoir une seule interface pour l'entraînement et la prédiction.
    """
    if features_type == "pixels":
        # Pixels bruts : on passe de (n, H, W) à (n, H*W)
        return images.reshape(images.shape[0], -1).astype(np.float32)

    if features_type == "hog":
        # HOG : on délègue à la fonction dédiée
        return extract_hog_features(
            images,
            target_size=target_size,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
        )

    raise ValueError('features_type doit être "pixels" ou "hog"')
