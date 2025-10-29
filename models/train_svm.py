# models/train_svm.py
import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import argparse
import numpy as np

def train_svm(feature_file, output_name):
    print(f"🔹 Cargando características desde {feature_file}")
    X, y, le = joblib.load(feature_file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("🔹 Entrenando SVM...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Precisión: {acc:.4f}")

    print("🔹 Guardando modelo...")
    os.makedirs("models/saved_models", exist_ok=True)
    joblib.dump((model, le), f"models/saved_models/{output_name}.pkl")

    print("🔹 Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True, choices=["hog", "lbp"])
    args = parser.parse_args()

    feature_path = f"features/{args.features}_features.pkl"
    output_name = f"{args.features}_svm"
    train_svm(feature_path, output_name)
