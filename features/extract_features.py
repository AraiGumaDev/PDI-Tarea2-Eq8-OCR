# features/extract_features.py
import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
import joblib

#Configuración
DATA_DIR = "data/train"
OUTPUT_FEATURES_DIR = "features/"
HOG_PARAMS = {'orientations': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2), 'block_norm': 'L2-Hys'}

# Número de clústeres para el modelo de Bag of Visual Words (SIFT)
NUM_CLUSTERS = 100

# Carga de imágenes desde el dataset
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

# Extracción de características HOG
def extract_hog_features(images):
    """
    Calcula las características HOG (Histogram of Oriented Gradients)
    para un conjunto de imágenes.
    """
    features = []
    for img in images:
        hog_feat = hog(img, **HOG_PARAMS)
        features.append(hog_feat)
    return np.array(features)


def extract_hog_features_training(image):
    """
    Calcula las características HOG de una sola imagen (uso durante predicción).
    """
    features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    return features


#Extracción de descriptores SIFT
def extract_sift_descriptors(images):
    """
    Extrae descriptores locales SIFT (Scale-Invariant Feature Transform)
    de todas las imágenes del conjunto.
    """
    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
    return descriptors_list

# Construcción del modelo Bag of Visual Words (BoVW)
def build_bovw(descriptor_list, num_clusters=NUM_CLUSTERS):
    """Construye el modelo de Bag of Visual Words usando KMeans."""
    all_descriptors = np.vstack(descriptor_list)
    print(f"Entrenando KMeans con {len(all_descriptors)} descriptores totales...")
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(all_descriptors)
    return kmeans

# Conversión de descriptores SIFT a histogramas BoVW
def extract_sift_features(images, kmeans_model):
    """
    Convierte los descriptores SIFT de cada imagen en un histograma
    basado en el modelo KMeans (BoVW). Cada histograma representa la
    frecuencia de aparición de los "visual words" en la imagen.
    """
    sift = cv2.SIFT_create()
    features = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is None:
            hist = np.zeros(kmeans_model.n_clusters)
        else:
            clusters = kmeans_model.predict(descriptors)
            hist, _ = np.histogram(clusters, bins=np.arange(0, kmeans_model.n_clusters + 1))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
        features.append(hist)
    return np.array(features)

# Guardado de características extraídas
def save_features(X, y, feature_type):
    os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump((X, y_enc, le), os.path.join(OUTPUT_FEATURES_DIR, f"{feature_type}_features.pkl"))
    print(f" Características {feature_type.upper()} guardadas en features/{feature_type}_features.pkl")

def main():
    print("Cargando imágenes de entrenamiento...")
    images, labels = load_images_from_folder(DATA_DIR)
    print(f"Total imágenes: {len(images)}")

    print("🔹 Extrayendo características HOG...")
    X_hog = extract_hog_features(images)
    save_features(X_hog, labels, "hog")

    print("Extrayendo descriptores SIFT...")
    descriptor_list = extract_sift_descriptors(images)

    print("🔹 Construyendo modelo Bag of Visual Words (BoVW)...")
    kmeans_model = build_bovw(descriptor_list, num_clusters=NUM_CLUSTERS)
    joblib.dump(kmeans_model, os.path.join(OUTPUT_FEATURES_DIR, "sift_kmeans.pkl"))
    print("Modelo KMeans guardado en features/sift_kmeans.pkl")

    print("🔹 Generando histogramas SIFT-BoVW...")
    X_sift = extract_sift_features(images, kmeans_model)
    save_features(X_sift, labels, "sift")

    print("Extracción completada.")



#Main
if __name__ == "__main__":
    print("Cargando imágenes de entrenamiento...")
    images, labels = load_images_from_folder(DATA_DIR)
    print(f"Total imágenes: {len(images)}")

    print("🔹 Extrayendo características HOG...")
    X_hog = extract_hog_features(images)
    save_features(X_hog, labels, "hog")

    print("Extrayendo descriptores SIFT...")
    descriptor_list = extract_sift_descriptors(images)

    print("Construyendo modelo Bag of Visual Words (BoVW)...")
    kmeans_model = build_bovw(descriptor_list, num_clusters=NUM_CLUSTERS)
    joblib.dump(kmeans_model, os.path.join(OUTPUT_FEATURES_DIR, "sift_kmeans.pkl"))
    print("Modelo KMeans guardado en features/sift_kmeans.pkl")

    print("Generando histogramas SIFT-BoVW...")
    X_sift = extract_sift_features(images, kmeans_model)
    save_features(X_sift, labels, "sift")

    print("Extracción completada.")
