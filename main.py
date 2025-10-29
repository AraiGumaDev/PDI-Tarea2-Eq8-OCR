#main.py
from utils.detect_plate import detect_plate
from utils.segment_characters import segment_characters
from utils.predict_plate import predict_characters
import numpy as np
import os

if __name__ == "__main__":
    img_path = "data/test/test7.jpg" #Ruta de la imagen a identificar la placa
    method = input("Inserte el modelo para detectar la matricula ( hog | sift ): ")
    detect_plate(img_path)
    char_paths = segment_characters("outputs/detected_plate.jpg")
    if method.upper() == "HOG":
        predict_characters("outputs/detected_plate.jpg", "hog", char_paths)
    elif method.upper() == "SIFT":
        predict_characters("outputs/detected_plate.jpg", "sift", char_paths)
    else :
        print("No selecciono un modelo valido para identificar la matricula (hog | sift )")
