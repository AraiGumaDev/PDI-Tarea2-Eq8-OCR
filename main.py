# main.py
from fontTools.misc.cython import returns

from utils.detect_plate import detect_plate
from utils.segment_characters import segment_characters
from features.extract_features import extract_hog_features
import joblib
import cv2
import os
import numpy as np


def predict_characters(plate_path, method, char_paths):
    recognized = ""

    if method.upper() == "HOG":
        model, le = joblib.load("models/saved_models/hog_svm.pkl")

        for c_path in char_paths:
            img = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            feat = extract_hog_features([img])
            pred = model.predict(feat)
            label = le.inverse_transform(pred)[0]
            recognized += clean_label(label)

        print(f"✅ Matrícula detectada: {recognized}")

        return recognized

    elif method.upper() == "LBP":
        print("Selecciono LBP")

        return "LBP"
    print ("No se seleccionono metodo")
    return recognized

def clean_label(label: str) -> str:
    if isinstance(label, str) and label.startswith("class_"):
        return label.replace("class_", "")
    return label

if __name__ == "__main__":
    img_path = "data/test/test3.jpeg"
    detect_plate(img_path)
    char_paths = segment_characters("outputs/detected_plate.jpg")
    predict_characters("outputs/detected_plate.jpg", "hog", char_paths)

