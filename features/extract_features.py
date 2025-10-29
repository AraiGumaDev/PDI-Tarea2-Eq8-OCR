# features/extract_features.py
import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import LabelEncoder
import joblib

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
DATA_DIR = "data/train"
OUTPUT_FEATURES_DIR = "features/"
HOG_PARAMS = {'orientations': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2), 'block_norm': 'L2-Hys'}
LBP_PARAMS = {'radius': 1, 'n_points': 8, 'method': 'uniform'}

# ----------------------------
# FUNCIONES
# ----------------------------
def load_images_from_folder(base_path):
    images, labels = [], []
    for label_folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, label_folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(folder_path, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (64, 64))
                images.append(img)
                labels.append(label_folder)
    return np.array(images), np.array(labels)


def extract_hog_features(images):
    features = []
    for img in images:
        hog_feat = hog(img, **HOG_PARAMS)
        features.append(hog_feat)
    return np.array(features)

def extract_hog_features_training(image):
    features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    return features

def extract_lbp_features(images):
    features = []
    for img in images:
        lbp = local_binary_pattern(img, LBP_PARAMS['n_points'], LBP_PARAMS['radius'], LBP_PARAMS['method'])
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_PARAMS['n_points'] + 3), range=(0, LBP_PARAMS['n_points'] + 2))
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-7
        features.append(hist)
    return np.array(features)

def extract_lbp_features_training(image, P=8, R=1):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P + 3),
                             range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


def save_features(X, y, feature_type):
    os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump((X, y_enc, le), os.path.join(OUTPUT_FEATURES_DIR, f"{feature_type}_features.pkl"))
    print(f"✅ Características {feature_type.upper()} guardadas en features/{feature_type}_features.pkl")


if __name__ == "__main__":
    print("🔹 Cargando imágenes de entrenamiento...")
    images, labels = load_images_from_folder(DATA_DIR)
    print(f"Total imágenes: {len(images)}")

    print("🔹 Extrayendo características HOG...")
    X_hog = extract_hog_features(images)
    save_features(X_hog, labels, "hog")

    print("🔹 Extrayendo características LBP...")
    X_lbp = extract_lbp_features(images)
    save_features(X_lbp, labels, "lbp")

    print("✅ Extracción completada.")
