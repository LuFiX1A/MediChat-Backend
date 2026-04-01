import os
import shutil
import uuid
import tensorflow as tf
import numpy as np
import cv2
from fastapi import UploadFile
import time

# --- CONFIGURACIÓN ---
UPLOAD_DIR = "uploads"

def cleanup_old_images(max_age_seconds=86400): # 86400 seg = 24 horas
    """Elimina imágenes de la carpeta uploads que tengan más de X tiempo."""
    now = time.time()
    count = 0
    if not os.path.exists(UPLOAD_DIR):
        return
        
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        # Verificacion del archivo y su antigüedad
        if os.path.isfile(file_path):
            file_age = os.stat(file_path).st_mtime
            if (now - file_age) > max_age_seconds:
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    print(f"No se pudo borrar {filename}: {e}")
    
    if count > 0:
        print(f"🧹 Limpieza completada: {count} imágenes viejas eliminadas.")

def setup_storage():
    """
    Crea la carpeta de subidas si no existe. 
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    cleanup_old_images() # <--- Limpia al arrancar
    print(f"📁 Carpeta de almacenamiento lista.")

# --- CARGA DEL MODELO ---
# Carga el modelo globalmente al importar el servicio
try:
    model = tf.keras.models.load_model('modelo_medichat_v5.h5')
    print("✅ Modelo CNN V5 cargado y listo para predicciones")
except Exception as e:
    print(f"❌ Error crítico cargando el modelo .h5: {e}")
    model = None

async def process_burn_logic(file: UploadFile):
    """
    Guarda la foto físicamente y la analiza con la CNN V5.
    """
    # 1. Asegurar nombre único para evitar sobreescritura
    ext = file.filename.split(".")[-1]
    unique_name = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    # Lectura del contenido una vez
    contents = await file.read()
    
    # Se guarda el archivo en el disco
    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    # 2. Análisis con la Red Neuronal
    if model is None:
        return {
            "grado": 1, 
            "confianza": 0, 
            "url": f"/uploads/{unique_name}", 
            "error": "El modelo de IA no pudo cargarse en el servidor"
        }

    try:
        # Pre-procesamiento de la imagen (OpenCV)
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Ajuste al formato que espera el modelo (224x224, RGB, Normalizado)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224)) 
        img_array = img_resized / 255.0               
        img_batch = np.expand_dims(img_array, axis=0) 

        # --- PREDICCIÓN REAL ---
        predictions = model.predict(img_batch)
        probabilidad = float(np.max(predictions))
        grado_idx = int(np.argmax(predictions)) # 0=G1, 1=G2, 2=G3
        
        # --- LÓGICA DE UMBRAL (SEGURIDAD) ---
        UMBRAL_MINIMO = 0.70  # Exigimos al menos 70% de certeza
        
        if probabilidad < UMBRAL_MINIMO:
            return {
                "grado": 0, # Grado 0 indica que no es concluyente
                "confianza": round(probabilidad * 100, 2),
                "url": f"/uploads/{unique_name}",
                "detalle": "La imagen no muestra patrones claros de una quemadura o la confianza es muy baja.",
                "error": "Baja confianza en el análisis"
            }

        # Si supera el umbral, se asigna el grado real (1, 2 o 3)
        grado_real = grado_idx + 1

        # 3. Respuesta unificada para el Frontend
        return {
            "grado": grado_real,
            "confianza": round(probabilidad * 100, 2),
            "url": f"/uploads/{unique_name}",
            "detalle": f"Análisis de IA: Posible Quemadura Grado {grado_real}"
        }

    except Exception as e:
        print(f"⚠️ Error en análisis de IA: {e}")
        return {
            "grado": 1, 
            "confianza": 0, 
            "url": f"/uploads/{unique_name}", 
            "error": "Error técnico al procesar la imagen"
        }


