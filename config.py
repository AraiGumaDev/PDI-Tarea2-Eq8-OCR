import os

# Directorio base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpetas principales
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
UTILS_DIR = os.path.join(BASE_DIR, "utils")

# Rutas clave
DETECTED_PLATE_PATH = os.path.join(OUTPUTS_DIR, "detected_plate.jpg")
SEGMENTED_DIR = os.path.join(OUTPUTS_DIR, "segmented_chars")

# Crear carpetas si no existen
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(SEGMENTED_DIR, exist_ok=True)
