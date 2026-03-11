import os
import shutil
import uuid
import random
from fastapi import UploadFile

UPLOAD_DIR = "uploads"

def setup_storage():
    """Asegura que la carpeta de fotos exista al arrancar."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)

async def process_burn_logic(file: UploadFile):
    """Lógica central: guarda la foto y simula el análisis de la IA."""
    # 1. Generar nombre único
    ext = file.filename.split(".")[-1]
    unique_name = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    # 2. Guardar archivo
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. Respuesta simulada (Aquí el Integrante A conectará su modelo después)
    # Simulamos grados 1, 2 o 3
    mock_grado = random.randint(1, 3)
    mock_probabilidad = round(random.uniform(0.75, 0.98), 2)

    return {
        "grado": mock_grado,
        "probabilidad": mock_probabilidad,
        "url": f"/uploads/{unique_name}"
    }