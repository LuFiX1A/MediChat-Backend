import cv2
import numpy as np
import uuid
import os

# Extensiones permitidas según la instrucción del líder
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def validate_image(filename: str) -> bool:
    """
    Tarea: Verificar que la extensión sea .jpg, .jpeg, .png.
    """
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def get_unique_filename(filename: str) -> str:
    """
    Tarea: Usar uuid para devolver un nombre único.
    Esto evita que dos fotos con el mismo nombre choquen en el servidor.
    """
    ext = os.path.splitext(filename)[1].lower()
    unique_name = f"{uuid.uuid4()}{ext}"
    return unique_name

def preprocess_for_cnn(image_path: str):
    """
    Tarea: Usar OpenCV para leer, aplicar ecualización y redimensionar a 224x224.
    """
    # 1. Leer la imagen desde la ruta proporcionada
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen en {image_path}")

    # 2. Ecualización de Histograma (CLAHE) 
    # Usamos CLAHE para normalizar la iluminación (clave en tu proyecto de quemaduras)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    merged_channels = cv2.merge((cl, a_channel, b_channel))
    final_img = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)

    # 3. Redimensionar a 224x224 (Formato estándar para la CNN MobileNetV2)
    resized_img = cv2.resize(final_img, (224, 224), interpolation=cv2.INTER_AREA)

    # 4. Guardar la imagen procesada sobreescribiendo la original o creando una nueva
    # Aquí la sobreescribimos para que la que se quede en /uploads sea la limpia
    cv2.imwrite(image_path, resized_img)
    
    return image_path