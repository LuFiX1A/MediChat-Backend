import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pymongo import MongoClient
from dotenv import load_dotenv
import sys
# Esto le dice a Python: "Mira también en la carpeta donde estás parado"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.burn_service import setup_storage, process_burn_logic
from services.gemini_service import obtener_respuesta_gemini

# --- 2. CONFIGURACIÓN INICIAL ---
load_dotenv()

app = FastAPI(
    title="API del Chatbot de Telemedicina",
    description="Backend optimizado y modularizado."
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
if not MONGO_URI:
    raise ValueError("No se encontró la MONGO_URI en el archivo .env")

try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["VirtualMedDB"]
    collection = db["doctors"]
    mongo_client.admin.command('ping')
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
    grado_ia: Optional[int] = None  # <-- IMPORTANTE: Ahora l chat puede recibir el grado

class ChatOutput(BaseModel):
    respuesta: Dict[str, Any]

# --- 6. ENDPOINT DEL CHAT (Lógica Unificada) ---
@app.post("/chat", response_model=ChatOutput)
async def handle_chat(input: ChatInput):
    try:
        # PASO A: Leer doctores de MongoDB
        texto_doctores_mongo = ""
        cursor_doctores = collection.find({})
        for doc in cursor_doctores:
            mongo_id = str(doc.get("_id", ""))
            nombre = doc.get("nombre", "")
            apellido = doc.get("apellido", "")
            especialidad = doc.get("especialidad", "Medicina General")
            texto_doctores_mongo += f"- ID: {mongo_id} | Dr/a: {nombre} {apellido} | Esp: {especialidad}\n"

        # PASO B: Llamar al servicio de Gemini (Ahora él tiene toda la configuración)
        parsed_response = await get_chat_response(
            mensaje_usuario=input.mensaje,
            contexto_medico=input.contexto_medico,
            texto_doctores_mongo=texto_doctores_mongo,
            grado_ia=input.grado_ia
        )

        return ChatOutput(respuesta=parsed_response)

    except Exception as e:
        print(f"Error general en el endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 7. ENDPOINT DE ANÁLISIS DE IMAGEN ---
@app.post("/analyze-burn")
async def analyze_burn(file: UploadFile = File(...)):
    # Delegamos el guardado y la simulación al servicio
    return await process_burn_logic(file)

# --- 8. ENDPOINT DE PRUEBA ---
@app.get("/")
def read_root():
    return {"status": "Online", "database": "Connected"}