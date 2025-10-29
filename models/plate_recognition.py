import cv2
import numpy as np
import joblib
from utils.preprocess import preprocess_image
from features.extract_hog import extract_hog_features
from features.extract_lbp import extract_lbp_features

def segment_and_recognize_plate(image_path, model_path="model_hog.pkl", feature_type="hog"):
    # Cargar modelo
    model = joblib.load(model_path)

    # Leer y preprocesar la placa completa
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detectar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if 0.2 < aspect_ratio < 1.0 and h > 30 and w > 10:  # Filtros básicos
            rois.append((x, y, w, h))

    # Ordenar de izquierda a derecha
    rois = sorted(rois, key=lambda r: r[0])

    recognized = ""
    for (x, y, w, h) in rois:
        char_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(char_img, (64, 64))
        if feature_type == "hog":
            features = extract_hog_features(resized)
        else:
            features = extract_lbp_features(resized)
        pred = model.predict([features])[0]
        recognized += str(pred)

    return recognized
