import cv2
import os
import numpy as np

def segment_characters(plate_path, output_dir="outputs/characters"):
    os.makedirs(output_dir, exist_ok=True)

    # Leer la placa
    plate = cv2.imread(plate_path)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # 1️⃣ Reducción de ruido y mejora de contraste
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.equalizeHist(gray)

    # 2️⃣ Binarización adaptativa (mejor que Otsu para luz irregular)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 15
    )

    # 3️⃣ Operaciones morfológicas para limpiar
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4️⃣ Buscar contornos
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

    if not chars:
        print("⚠️ No se detectaron caracteres en la placa (ajusta parámetros).")
        return []

    # Ordenar de izquierda a derecha
    chars = sorted(chars, key=lambda c: c[0])
    saved_paths = []

    for i, (_, char) in enumerate(chars):
        # Añadir padding para que todos los caracteres queden centrados
        char = cv2.copyMakeBorder(char, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0,0,0])
        char_path = os.path.join(output_dir, f"char_{i}.png")
        cv2.imwrite(char_path, char)
        saved_paths.append(char_path)

    print(f"✅ {len(saved_paths)} caracteres segmentados y guardados en {output_dir}")
    return saved_paths

if __name__ == "__main__":
    segment_characters("outputs\detected_plate.jpg")

