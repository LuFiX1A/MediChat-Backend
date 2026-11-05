import os
import json
import sys
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, AliasChoices
from typing import List, Optional, Dict, Any
from pymongo import MongoClient
from dotenv import load_dotenv

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Esto asegura que Python encuentre las carpetas 'services' y 'api_modules'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.burn_service import setup_storage, process_burn_logic
from services.gemini_service import get_gemini_response, obtener_respuesta_gemini
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
db = None
doctors_collection = None
collection = None

try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["VirtualMedDB"]
    doctors_collection = db["doctors"]
    collection = doctors_collection  # Alias para compatibilidad hacia atrás
    print("✅ ¡Conexión exitosa a MongoDB Atlas!")
except Exception as e:
    print(f"❌ Error CRÍTICO conectando a Mongo: {e}")

# Preparar almacenamiento de fotos y montar estáticos
setup_storage()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# --- 5. MODELOS DE DATOS ---
class ChatInput(BaseModel):
    user_id: Optional[str] = "anonimo"
    message: str = Field(validation_alias=AliasChoices('message', 'mensaje'))
    medical_context: Optional[str] = Field(default="Ninguno", validation_alias=AliasChoices('medical_context', 'contexto_medico'))
    ai_grade: Optional[int] = Field(default=None, validation_alias=AliasChoices('ai_grade', 'grado_ia'))

    # Propiedades para compatibilidad de acceso por atributo en español
    @property
    def mensaje(self) -> str:
        return self.message

    @property
    def contexto_medico(self) -> Optional[str]:
        return self.medical_context

    @property
    def grado_ia(self) -> Optional[int]:
        return self.ai_grade

class ChatOutput(BaseModel):
    response: Dict[str, Any] = Field(default_factory=dict)
    respuesta: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context: Any) -> None:
        if self.respuesta is None and self.response:
            self.respuesta = self.response
        elif not self.response and self.respuesta:
            self.response = self.respuesta

# --- 6. ENDPOINT DEL CHAT (Solo Texto) ---
@app.post("/chat", response_model=ChatOutput)
async def handle_chat(chat_input: ChatInput):
    try:
        # Extraer doctores de Mongo para dar contexto a Gemini
        doctors_list = []
        if doctors_collection is not None:
            try:
                doctors_list = list(doctors_collection.find({}, {"_id": 0}))
            except Exception as mongo_err:
                print(f"Advertencia: No se pudieron obtener los doctores de MongoDB: {mongo_err}")
        
        parsed_response = await get_gemini_response(
            user_message=chat_input.message,
            medical_context=chat_input.medical_context,
            mongo_doctors_text=str(doctors_list),
            ai_grade=chat_input.ai_grade
        )
        return ChatOutput(response=parsed_response)
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
        doctors_list = []
        if doctors_collection is not None:
            try:
                doctors_list = list(doctors_collection.find({}, {"_id": 0}))
            except Exception as mongo_err:
                print(f"Advertencia: No se pudieron obtener los doctores de MongoDB: {mongo_err}")
        
        if file is None:
            # Caso: Solo texto (Chat general)
            ai_response = await get_gemini_response(
                user_message=user_text, 
                medical_context="Consulta Médica General", 
                mongo_doctors_text=str(doctors_list)
            )
            return {
                "visual_analysis": {"detail": "No se proporcionó imagen", "detalle": "No se proporcionó imagen"},
                "ai_diagnosis": ai_response,
                "user_text": user_text,
                # Claves en español para compatibilidad con el frontend
                "analisis_visual": {"detalle": "No se proporcionó imagen"},
                "diagnostico_ia": ai_response,
                "texto_usuario": user_text
            }
        else:
            # Caso: Con imagen (Usa el Handler que une CNN y Gemini)
            return await handle_integration_logic(file, user_text, doctors_list)

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