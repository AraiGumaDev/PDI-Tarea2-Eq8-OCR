# models/train_svm.py
import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import argparse
import numpy as np


def train_svm(feature_file, output_name, method, kmeans=None):
    """
    Entrena un modelo SVM utilizando las características extraídas (HOG o SIFT-BoVW).

    Parámetros:
        feature_file (str): Ruta al archivo .pkl con las características y etiquetas.
        output_name (str): Nombre base para guardar el modelo entrenado.
        method (str): Tipo de descriptor utilizado ('hog' o 'sift').
        kmeans (MiniBatchKMeans, opcional): Modelo KMeans usado en SIFT-BoVW.
    """
    print(f"Cargando características desde {feature_file}")
    X, y, le = joblib.load(feature_file)

    # División de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f" Entrenando SVM ({method.upper()})...")
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f" Precisión: {acc:.4f}")

    print("--Guardando modelo entrenado...--")
    os.makedirs("models/saved_svm", exist_ok=True)

    # Guardado distinto según método
    if method == "hog":
        # Guardamos como (model, label_encoder)
        joblib.dump((model, le), f"models/saved_svm/{output_name}.pkl")
    elif method == "sift":
        # Guardamos como (model, kmeans, label_encoder)
        if kmeans is None:
            print("Advertencia: No se recibió un modelo KMeans para SIFT.")
        joblib.dump((model, kmeans, le), f"models/saved_svm/{output_name}.pkl")

    # Reporte final
    print("---Matriz de confusión:---")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    """
    Permite ejecutar el entrenamiento desde consola usando:
        python train_svm.py --features [hog|sift]

    Según el argumento recibido:
        - Carga el archivo de características correspondiente.
        - Si es SIFT, también carga el modelo KMeans generado en extract_features.py.
        - Entrena el modelo SVM y lo guarda en la carpeta models/saved_svm.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True, choices=["hog", "sift"])
    args = parser.parse_args()

    feature_path = f"features/{args.features}_features.pkl"
    output_name = f"{args.features}_svm"

    if args.features == "sift":
        kmeans_path = "features/sift_kmeans.pkl"
        if os.path.exists(kmeans_path):
            print("Cargando modelo KMeans (SIFT-BoVW)...")
            kmeans = joblib.load(kmeans_path)
        else:
            kmeans = None
            print("No se encontró 'sift_kmeans.pkl'. Asegúrate de haber ejecutado extract_features.py antes.")
        train_svm(feature_path, output_name, method="sift", kmeans=kmeans)
    else:
        train_svm(feature_path, output_name, method="hog")
