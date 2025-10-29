import joblib
from utils.preprocess import preprocess_image
from features.extract_hog import extract_hog_features
from features.extract_lbp import extract_lbp_features

def predict_image(model_path, image_path, feature_type="hog"):
    model = joblib.load(model_path)
    img = preprocess_image(image_path)
    if feature_type == "hog":
        features = extract_hog_features(img)
    else:
        features = extract_lbp_features(img)
    prediction = model.predict([features])[0]
    return prediction
