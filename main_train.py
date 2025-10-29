#models/train_svm.py
import argparse

from features.extract_features import main as extract_features_main
from models.train_svm import train_svm


if __name__ == "__main__":
    data_dir = "data/train"

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True, choices=["hog", "sift"])
    args = parser.parse_args()

    feature_path = f"features/{args.features}_features.pkl"
    output_name = f"{args.features}_svm"

    extract_features_main()

    #Entrena el modelo seleccionado con --features
    if args.features == "hog":
        train_svm(feature_path, output_name, method="hog")
    elif args.features == "sift":
        kmeans_path = "features/sift_kmeans.pkl"
        train_svm(feature_path, output_name, method="sift", kmeans=kmeans_path)





