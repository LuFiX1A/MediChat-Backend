import os
import json
import sys
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pymongo import MongoClient
from dotenv import load_dotenv

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Esto asegura que Python encuentre las carpetas 'services' y 'api_modules'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.burn_service import setup_storage, process_burn_logic
from services.gemini_service import obtener_respuesta_gemini
from api_modules.burn_handler import handle_integration_logic

# --- 2. CONFIGURACIÓN INICIAL ---
load_dotenv()

app = FastAPI(
    title="MediChat API V5 - Producción",
    description="Backend con CNN real y triaje avanzado por Gemini."
)

# --- 3. CONFIGURACIÓN DE CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. CONEXIÓN A MONGODB ATLAS ---
MONGO_URI = os.getenv('MONGO_URI')
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["VirtualMedDB"]
    collection = db["doctors"]
    print("✅ ¡Conexión exitosa a MongoDB Atlas!")
except Exception as e:
    print(f"❌ Error CRÍTICO conectando a Mongo: {e}")

# Preparar almacenamiento de fotos y montar estáticos
setup_storage()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# --- 5. MODELOS DE DATOS ---
class ChatInput(BaseModel):
    user_id: Optional[str] = "anonimo"
    mensaje: str
    contexto_medico: Optional[str] = "Ninguno"
    grado_ia: Optional[int] = None

class ChatOutput(BaseModel):
    respuesta: Dict[str, Any]

# --- 6. ENDPOINT DEL CHAT (Solo Texto) ---
@app.post("/chat", response_model=ChatOutput)
async def handle_chat(input: ChatInput):
    try:
        # Extraer doctores de Mongo para dar contexto a Gemini
        doctores_list = list(collection.find({}, {"_id": 0}))
        
        parsed_response = await obtener_respuesta_gemini(
            mensaje_usuario=input.mensaje,
            contexto_medico=input.contexto_medico,
            texto_doctores_mongo=str(doctores_list),
            grado_ia=input.grado_ia
        )
        return ChatOutput(respuesta=parsed_response)
    except Exception as e:
        print(f"Error en endpoint /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 7. ENDPOINT DE ANÁLISIS DE IMAGEN (Solo CNN) ---
@app.post("/analyze-burn")
async def analyze_burn(file: UploadFile = File(...)):
    """
    Endpoint para probar exclusivamente la clasificación de la CNN V5.
    Útil para diagnosticar errores de visión sin llamar a Gemini.
    """
    try:
        return await process_burn_logic(file)
    except Exception as e:
        print(f"Error en analyze-burn: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 8. ENDPOINT DE INTEGRACIÓN TOTAL (Imagen + Texto + Gemini) ---
@app.post("/analyze-full")
async def analyze_full(
    file: Optional[UploadFile] = File(None), 
    user_text: str = Form(...)
):
    """
    Endpoint principal para la App Ionic. Procesa imagen y genera respuesta médica.
    """
    try:
        # 1. Obtenemos los doctores (Síncrono para evitar errores de tipo)
        doctores_list = list(db["doctors"].find({}, {"_id": 0}))
        
        if file is None:
            # Caso: Solo texto (Chat general)
            respuesta = await obtener_respuesta_gemini(
                mensaje_usuario=user_text, 
                contexto_medico="Consulta Médica General", 
                texto_doctores_mongo=str(doctores_list)
            )
            return {
                "analisis_visual": {"detalle": "No se proporcionó imagen"},
                "diagnostico_ia": respuesta,
                "texto_usuario": user_text
            }
        else:
            # Caso: Con imagen (Usa el Handler que une CNN y Gemini)
            return await handle_integration_logic(file, user_text, doctores_list)

    except Exception as e:
        print(f"Error detallado en analyze-full: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 9. ENDPOINT DE PRUEBA ---
@app.get("/")
def read_root():
    return {
        "status": "Online", 
        "model": "CNN-V5 Ready", 
        "database": "Connected",
        "api_version": "2.0.0"
    }