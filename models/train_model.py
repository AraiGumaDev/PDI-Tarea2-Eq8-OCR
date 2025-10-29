import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from utils.preprocess import preprocess_image
from features.extract_hog import extract_hog_features
from features.extract_lbp import extract_lbp_features

def load_dataset(data_dir, feature_type="hog"):
    X, y = [], []
    for label in os.listdir(data_dir):
        class_path = os.path.join(data_dir, label)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = preprocess_image(img_path)
            if feature_type == "hog":
                features = extract_hog_features(img)
            elif feature_type == "lbp":
                features = extract_lbp_features(img)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

def train_and_save_model(data_dir, feature_type="hog", model_path="model.pkl"):
    X, y = load_dataset(data_dir, feature_type)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"Modelo {feature_type.upper()} guardado en: {model_path}")
