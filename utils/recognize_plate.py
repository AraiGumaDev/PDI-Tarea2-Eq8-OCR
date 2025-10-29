import joblib
import cv2
from skimage.feature import hog

def extract_hog(img):
    return hog(cv2.resize(img, (64, 64)), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

def recognize_plate(char_paths, model_path="models/model_hog.pkl"):
    model = joblib.load(model_path)
    result = ""

    for path in char_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        features = extract_hog(img).reshape(1, -1)
        pred = model.predict(features)[0]
        result += str(pred)

    print(f"✅ Matrícula reconocida: {result}")
    return result

if __name__ == "__main__":
    recognize_plate(["../outputs/characters/char_0.png"])
