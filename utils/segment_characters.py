import shutil

import cv2
import os
import numpy as np

from utils.detect_plate import detect_plate


def segment_characters(plate_path, output_dir="outputs/characters"):
    os.makedirs(output_dir, exist_ok=True)

    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
    print(f"Directory '{output_dir}' has been emptied.")

    # Leer la placa
    plate = cv2.imread(plate_path)
    if plate is None:
        print(f"Plate is not found in{plate_path}")
        return

    #Agregandole cualquier otro filtro se obtienen peores resultados
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (3, 3), 0)
    #gray = cv2.bilateralFilter(gray, 11, 17, 17)
    #gray = cv2.equalizeHist(gray)
    #gray = cv2.filter2D(gray, -1, gray)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 15
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    h_plate, w_plate = thresh.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h

        # Filtrar por tamaño y proporción típica de caracteres
        if 0.2 < aspect_ratio < 1.0 and 0.01*h_plate*w_plate < area < 0.2*h_plate*w_plate:
            char_img = thresh[y:y+h, x:x+w]
            chars.append((x, char_img))

    test_path = os.path.join(output_dir, "test.png")
    cv2.imwrite(test_path, thresh)

    if not chars:
        print("⚠️ No se detectaron caracteres en la placa (ajusta parámetros).")
        return []


    # Ordenar de izquierda a derecha
    chars = sorted(chars, key=lambda c: c[0])
    saved_paths = []

    for i, (_, char) in enumerate(chars):
        # Añadir padding para que todos los caracteres queden centrados
        char = cv2.copyMakeBorder(char, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0,0,0])
        char = cv2.resize(char, (64, 64))
        char_path = os.path.join(output_dir, f"char_{i}.png")
        cv2.imwrite(char_path, char)
        saved_paths.append(char_path)

    print(f"✅ {len(saved_paths)} caracteres segmentados y guardados en {output_dir}")
    return saved_paths

if __name__ == "__main__":
    #segment_characters("M:\\Users\\mahyro\\Documents\\Universidad\\2025-2\\Procesamiento Digital de Imagenes\\Tarea 2\\matricula_ocr\\outputs\\detected_plate.jpg")
    segment_characters("../outputs/detected_plate.jpg")
