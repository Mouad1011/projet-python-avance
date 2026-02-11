import argparse

from src.digit_classifier.train import train_and_save
from src.digit_classifier.evaluate import evaluate_model
from src.digit_classifier.predict import predict_by_index


def main():
    """
    Interface en ligne de commande pour entraîner, évaluer
    et tester le classifieur de chiffres manuscrits.
    """
    parser = argparse.ArgumentParser(
        description="CLI pour la classification de chiffres manuscrits"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ----- Entraînement -----
    subparsers.add_parser(
        "train",
        help="Entraîner le modèle et le sauvegarder"
    )

    # ----- Évaluation -----
    subparsers.add_parser(
        "evaluate",
        help="Évaluer le modèle entraîné"
    )

    # ----- Prédiction -----
    predict_parser = subparsers.add_parser(
        "predict",
        help="Prédire un chiffre à partir de son index"
    )
    predict_parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Index de l'image à prédire dans le dataset"
    )

    args = parser.parse_args()

    if args.command == "train":
        accuracy = train_and_save()
        print(f"Entraînement terminé. Accuracy : {accuracy:.3f}")

    elif args.command == "evaluate":
        accuracy, img_path = evaluate_model()
        print(f"Évaluation terminée. Accuracy : {accuracy:.3f}")
        print(f"Matrice de confusion sauvegardée dans : {img_path}")

    elif args.command == "predict":
        prediction, true_label = predict_by_index(args.index)
        print(f"Prédiction : {prediction} | Label réel : {true_label}")


if __name__ == "__main__":
    main()
