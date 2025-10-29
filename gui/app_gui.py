# gui/app_gui.py
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from main import predict_characters
from utils.detect_plate import detect_plate

class LicensePlateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Matrículas - India OCR")
        self.root.geometry("800x600")

        self.label = tk.Label(root, text="Selecciona una imagen de vehículo", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn = tk.Button(root, text="Cargar imagen", command=self.load_image)
        self.btn.pack(pady=5)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.result_label = tk.Label(root, text="", font=("Arial", 16))
        self.result_label.pack(pady=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.png *.jpeg")])
        if not file_path:
            return

        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

        plate_path = detect_plate(file_path)
        result = predict_characters(plate_path)
        self.result_label.config(text=f"Matrícula detectada: {result}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateGUI(root)
    root.mainloop()
