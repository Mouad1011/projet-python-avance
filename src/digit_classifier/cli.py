import argparse

from src.digit_classifier.train import train_and_save
from src.digit_classifier.evaluate import evaluate_model
from src.digit_classifier.predict import predict_by_index


def main():
    parser = argparse.ArgumentParser(
        description="CLI pour la classification de chiffres manuscrits (baseline + version avancée)"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ----- options communes train/evaluate -----
    def add_common_args(p):
        p.add_argument(
            "--features",
            choices=["pixels", "hog"],
            default="pixels",
            help="Type de représentation : pixels (baseline) ou hog (avancé)"
        )
        p.add_argument(
            "--model",
            choices=["logreg", "svm"],
            default="logreg",
            help="Type de modèle : logreg (baseline) ou svm (avancé)"
        )

    # ----- train -----
    train_parser = subparsers.add_parser("train", help="Entraîner le modèle et le sauvegarder")
    add_common_args(train_parser)

    # ----- evaluate -----
    eval_parser = subparsers.add_parser("evaluate", help="Évaluer le modèle entraîné")
    add_common_args(eval_parser)

    # ----- predict -----
    predict_parser = subparsers.add_parser("predict", help="Prédire un chiffre à partir de son index")
    predict_parser.add_argument("--index", type=int, required=True, help="Index de l'image à prédire")
    predict_parser.add_argument(
        "--model-path",
        default="models/digit_model_pixels_logreg.joblib",
        help="Chemin vers le modèle à charger (par défaut : baseline)"
    )

    args = parser.parse_args()

    if args.command == "train":
        acc, path = train_and_save(features_type=args.features, model_type=args.model)
        print(f"Entraînement terminé. Accuracy : {acc:.3f}")
        print(f"Modèle sauvegardé dans : {path}")

    elif args.command == "evaluate":
        acc, img = evaluate_model(features_type=args.features, model_type=args.model)
        print(f"Évaluation terminée. Accuracy : {acc:.3f}")
        print(f"Matrice de confusion sauvegardée dans : {img}")

    elif args.command == "predict":
        pred, true = predict_by_index(args.index, model_path=args.model_path)
        print(f"Prédiction : {pred} | Label réel : {true}")


if __name__ == "__main__":
    main()
