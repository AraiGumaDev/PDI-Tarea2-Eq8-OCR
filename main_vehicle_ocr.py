from utils.detect_plate import detect_plate
from utils.segment_characters import segment_characters
from utils.recognize_plate import recognize_plate

def main(image_path):
    print("\n🚗 Iniciando pipeline de reconocimiento de matrícula...\n")

    # 1️⃣ Detección de la placa
    plate_path = detect_plate(image_path)
    if not plate_path:
        return

    # 2️⃣ Segmentación de caracteres
    char_paths = segment_characters(plate_path)
    if not char_paths:
        return

    # 3️⃣ Reconocimiento
    plate_number = recognize_plate(char_paths)
    print(f"\n🔹 Resultado final: {plate_number}\n")

if __name__ == "__main__":
    main("data/test/car3.jpeg")
