# models/evaluate_model.py
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

def evaluate_model(feature_file, model_file):
    X, y, le = joblib.load(feature_file)
    model, le_model = joblib.load(model_file)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"✅ Accuracy global: {acc:.4f}")
    print("Matriz de confusión:")
    print(confusion_matrix(y, y_pred))
    print("Reporte de clasificación:")
    print(classification_report(y, y_pred, target_names=le.classes_))


if __name__ == "__main__":
    evaluate_model("features/lbp_features.pkl", "models/saved_svm/lbp_svm.pkl")
