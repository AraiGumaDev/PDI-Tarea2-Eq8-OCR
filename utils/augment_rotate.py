# augment_rotate.py
import os
import cv2
import numpy as np

# Ruta base de las carpetas
BASE_DIR = "data/train"

# Ángulos de rotación
ROTATION_ANGLES = [10, -10]  # puedes ajustarlos si quieres más variación

# Tamaño original
IMG_SIZE = (28, 28)


def rotate_image_keep_size(image, angle):
    """Rota una imagen sin cortar el contenido, manteniendo el tamaño original."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Obtener la matriz de rotación
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calcular los nuevos límites para evitar recortes
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Ajustar la matriz para el nuevo centro
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    # Aplicar rotación
    rotated = cv2.warpAffine(image, rot_mat, (new_w, new_h), borderValue=(0, 0, 0))

    # Redimensionar de nuevo a 28x28
    rotated_resized = cv2.resize(rotated, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return rotated_resized


def augment_dataset(base_dir):
    """Genera imágenes rotadas para todas las clases en la carpeta base."""
    total_new = 0

    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"🔹 Procesando clase: {class_name}")

        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            # Aplicar rotaciones y guardar copias
            for i, angle in enumerate(ROTATION_ANGLES):
                rotated = rotate_image_keep_size(img, angle)

                # Nombre nuevo archivo
                name, ext = os.path.splitext(img_name)
                new_name = f"{name}_rot{i+1}{ext}"
                new_path = os.path.join(class_path, new_name)

                cv2.imwrite(new_path, rotated)
                total_new += 1

    print(f"Aumento completado. Imágenes nuevas creadas: {total_new}")


if __name__ == "__main__":
    augment_dataset(BASE_DIR)
