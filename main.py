# main.py
from utils.detect_plate import detect_plate
from utils.segment_characters import segment_characters
from features.extract_features import extract_hog_features
import joblib
import cv2
import numpy as np
import os


def predict_characters(plate_path, method, char_paths):
    recognized = ""

    if method.upper() == "HOG":
        # Modelo HOG + SVM
        model, le = joblib.load("models/saved_svm/hog_svm.pkl")

        for c_path in char_paths:
            img = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            feat = extract_hog_features([img])
            pred = model.predict(feat)
            label = le.inverse_transform(pred)[0]
            recognized += clean_label(label)

        print(f"✅ Matrícula detectada (HOG): {recognized}")
        return recognized

    elif method.upper() == "SIFT":
        # Modelos SIFT + KMeans + SVM
        model, kmeans, le = joblib.load("models/saved_svm/sift_svm.pkl")

        sift = cv2.SIFT_create()

        for c_path in char_paths:
            img = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))  # tamaño más grande ayuda a SIFT

            # Extraer descriptores SIFT
            keypoints, descriptors = sift.detectAndCompute(img, None)

            if descriptors is None:
                # Si no detecta puntos clave, usar vector nulo
                hist = np.zeros((kmeans.n_clusters,), dtype=np.float32)
            else:
                # Asignar cada descriptor al cluster más cercano
                clusters = kmeans.predict(descriptors)
                hist, _ = np.histogram(clusters, bins=np.arange(kmeans.n_clusters + 1))
                hist = hist.astype(np.float32)
                hist /= (hist.sum() + 1e-7)  # normalización

            # Predecir con el SVM
            hist = hist.reshape(1, -1)
            pred = model.predict(hist)
            label = le.inverse_transform(pred)[0]
            recognized += clean_label(label)

        print(f"✅ Matrícula detectada (SIFT): {recognized}")
        return recognized

    print("⚠️ No se seleccionó un método válido")
    return "No se seleccionó un método válido"


def clean_label(label: str) -> str:
    if isinstance(label, str) and label.startswith("class_"):
        return label.replace("class_", "")
    return label


if __name__ == "__main__":
    img_path = "data/test/test3.jpeg"
    detect_plate(img_path)
    char_paths = segment_characters("outputs/detected_plate.jpg")
    predict_characters("outputs/detected_plate.jpg", "sift", char_paths)
