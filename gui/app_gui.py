import os
import cv2
import joblib
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import sys
from io import StringIO

# Importar tus funciones
from main import predict_characters
from utils.detect_plate import detect_plate
from utils.segment_characters import segment_characters
from customtkinter import CTkImage

# Estilo general
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class RedirectText:
    """Permite redirigir sys.stdout a un Textbox de customtkinter"""
    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, string):
        self.textbox.configure(state="normal")
        self.textbox.insert("end", string)
        self.textbox.see("end")
        self.textbox.configure(state="disabled")

    def flush(self):
        pass  # Necesario para compatibilidad con sys.stdout

class LicensePlateApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Detector de Placas | HOG / LBP")
        self.geometry("1100x600")
        self.resizable(False, False)

        self.image_path = None
        self.current_image = None
        self.selected_method = ctk.StringVar(value="HOG")

        # Cargar modelos
        self.model_hog = self.load_model("models/saved_svm/hog_svm.pkl")
        self.model_lbp = self.load_model("models/saved_svm/lbp_svm.pkl")

        # Crear la interfaz
        self.create_layout()

    def load_model(self, path):
        if os.path.exists(path):
            print(f"[INFO] Modelo cargado: {path}")
            return joblib.load(path)
        else:
            print(f"[WARN] No se encontró el modelo en {path}")
            return None

    def create_layout(self):
        # === PANEL IZQUIERDO ===
        sidebar = ctk.CTkFrame(self, width=250, corner_radius=10)
        sidebar.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(sidebar, text="Detector de Placas", font=("Arial", 20, "bold")).pack(pady=20)

        ctk.CTkButton(sidebar, text="Cargar imagen", command=self.load_image).pack(pady=10)

        ctk.CTkLabel(sidebar, text="Método de características:", font=("Arial", 14)).pack(pady=(20, 5))
        ctk.CTkOptionMenu(sidebar, variable=self.selected_method, values=["HOG", "LBP"]).pack(pady=10)

        ctk.CTkButton(sidebar, text="Detectar Placa", command=self.process_detection).pack(pady=20)

        ctk.CTkLabel(sidebar, text="Texto detectado:", font=("Arial", 14)).pack(pady=(20, 5))
        self.result_label = ctk.CTkLabel(sidebar, text="", font=("Consolas", 16, "bold"))
        self.result_label.pack(pady=10)

        # === PANEL CENTRAL (IMAGEN) ===
        self.image_panel = ctk.CTkLabel(self, text="")
        self.image_panel.pack(side="left", expand=True, padx=10, pady=10)

        # === PANEL DERECHO (CONSOLA) ===
        console_frame = ctk.CTkFrame(self, width=300, corner_radius=10)
        console_frame.pack(side="right", fill="both", padx=10, pady=10)

        ctk.CTkLabel(console_frame, text="Consola del proceso", font=("Arial", 14, "bold")).pack(pady=10)
        self.console = ctk.CTkTextbox(console_frame, width=280, height=500, corner_radius=10)
        self.console.pack(fill="both", expand=True, padx=5, pady=5)
        self.console.configure(state="disabled")

        # Redirigir stdout
        sys.stdout = RedirectText(self.console)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.webp")])
        if not file_path:
            return

        self.image_path = file_path
        image = Image.open(file_path)
        image.thumbnail((600, 500))
        self.current_image = np.array(image)
        self.display_image(image)
        self.result_label.configure(text="")
        print(f"[INFO] Imagen cargada: {file_path}")

    def display_image(self, image):
        image = image.convert("RGB")
        ctk_image = CTkImage(light_image=image, dark_image=image, size=image.size)
        self.image_panel.configure(image=ctk_image)
        self.image_panel.image = ctk_image

    def process_detection(self):
        if not self.image_path:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        try:
            print(f"\n[INFO] Detectando placa con método {self.selected_method.get()}...")
            detected_plate_image = detect_plate(self.image_path)

            # Mostrar la placa detectada
            detected_img = Image.open("outputs/detected_plate_box.jpg")
            detected_img.thumbnail((600, 500))
            self.display_image(detected_img)

            # Segmentar caracteres
            char_paths = segment_characters("outputs/detected_plate.jpg")
            print(f"[INFO] Caracteres segmentados: {len(char_paths)}")

            # Predicción según modelo seleccionado
            method = self.selected_method.get()
            if method == "HOG" and self.model_hog:
                text = predict_characters("outputs/detected_plate.jpg", "HOG", char_paths)
            elif method == "LBP" and self.model_lbp:
                text = predict_characters("outputs/detected_plate.jpg", "LBP", char_paths)
            else:
                print("[ERROR] Modelo no cargado.")
                return

            print(f"[RESULTADO] Texto detectado: {text}")
            self.result_label.configure(text=text)

        except Exception as e:
            print(f"[ERROR] {e}")
            messagebox.showerror("Error", f"Ocurrió un error:\n{e}")

if __name__ == "__main__":
    app = LicensePlateApp()
    app.mainloop()
