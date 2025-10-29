import cv2
import os

def detect_plate(image_path, save_path="outputs/detected_plate.jpg"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Leer imagen y convertir a escala de grises
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Clasificador Haar preentrenado
    cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    plate_cascade = cv2.CascadeClassifier(cascade_path)

    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(25, 25))

    if len(plates) == 0:
        print("⚠️ No se detectó ninguna placa en la imagen.")
        return None

    img_box = img.copy()

    for plate in plates:
        (x, y, w, h) = plate
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img_box[y:y + h, x:x + w]
        cv2.rectangle(img_box, (x, y), (x + w, y + h), (255, 0, 0), 2)

    x, y, w, h = plates[0]
    plate_img = img[y:y+h, x:x+w]
    cv2.imwrite(save_path, plate_img)
    print(f"✅ Placa detectada y guardada en {save_path}")
    test_path = os.path.join("outputs/", "detected_plate_box.jpg")
    cv2.imwrite(test_path, img_box)
    return img

