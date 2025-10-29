# models/evaluate_model.py
import argparse
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(feature_file, model_file, method):
    """
    Evalúa el modelo SVM entrenado (HOG o SIFT) calculando precisión, matriz de confusión y reporte de clasificación.
    Además, muestra una gráfica de la matriz de confusión en pantalla.

    Parámetros:
        feature_file (str): Ruta al archivo .pkl con las características y etiquetas codificadas.
        model_file (str): Ruta al archivo .pkl con el modelo entrenado.
        method (str): Método de extracción de características ('hog' o 'sift').
    """

    # Carga de características y etiquetas
    print(f"Cargando características desde: {feature_file}")
    X, y, le = joblib.load(feature_file)

    # Carga del modelo (estructura distinta para HOG y SIFT)
    print(f"Cargando modelo desde: {model_file}")
    model_data = joblib.load(model_file)

    if method == "hog":
        model, le_model = model_data
    elif method == "sift":
        # En SIFT se guardó como (model, kmeans, label_encoder)
        model, kmeans, le_model = model_data
    else:
        raise ValueError("Método no válido. Usa 'hog' o 'sift'.")

    # Predicciones
    print("Realizando predicciones...")
    y_pred = model.predict(X)

    # Evaluación
    acc = accuracy_score(y, y_pred)
    print(f"\nPrecisión global: {acc:.4f}\n")
    print("Matriz de confusión:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    print("\nReporte de clasificación:")
    print(classification_report(y, y_pred, target_names=le.classes_))

    # Visualización de la matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.title(f"Matriz de Confusión ({method.upper()}) - Precisión: {acc:.4f}")
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["hog", "sift"])
    args = parser.parse_args()

    if args.model == "hog":
        evaluate_model(
            "features/hog_features.pkl",
            "models/saved_svm/hog_svm.pkl",
            method="hog"
        )
    elif args.model == "sift":
        evaluate_model(
            "features/sift_features.pkl",
            "models/saved_svm/sift_svm.pkl",
            method="sift"
        )
