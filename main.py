# main.py
from utils.detect_plate import detect_plate
from utils.segment_characters import segment_characters
from features.extract_features import extract_hog_features
import joblib
import cv2
import os
import numpy as np

MODEL_PATH = "models/saved_models/hog_svm.pkl"

def predict_characters(plate_path):
    print("🔹 Segmentando caracteres...")
    char_paths = segment_characters(plate_path)
    model, le = joblib.load(MODEL_PATH)

    recognized = ""
    for c_path in char_paths:
        img = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        feat = extract_hog_features([img])
        pred = model.predict(feat)
        recognized += le.inverse_transform(pred)[0]

    print(f"✅ Matrícula detectada: {recognized}")
    return recognized


if __name__ == "__main__":
    img_path = "data/test/testing/12.jpg"
    plate_path = detect_plate(img_path)
    predict_characters(plate_path)
