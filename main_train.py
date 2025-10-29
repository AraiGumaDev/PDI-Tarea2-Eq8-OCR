from models.train_model import train_and_save_model

if __name__ == "__main__":
    data_dir = "data/train"
    # Entrena dos modelos, uno con HOG y otro con LBP
    train_and_save_model(data_dir, feature_type="hog", model_path="model_hog.pkl")
    train_and_save_model(data_dir, feature_type="lbp", model_path="model_lbp.pkl")


