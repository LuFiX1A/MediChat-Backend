import os
import sys
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, AliasChoices
from typing import Optional, Dict, Any
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.nlp_service import analyze_clinical_note, get_nlp_evaluation_metrics
from services.scheduler_service import optimize_appointment_schedule
from database.hybrid_db import check_hybrid_health, doctors_col
from services.burn_service import setup_storage, process_burn_logic
from services.gemini_service import get_gemini_response
from api_modules.burn_handler import handle_integration_logic
from routes_healthtext_v2 import router as healthtext_v2_router, ensure_v2_schema_and_sync

load_dotenv()

app = FastAPI(
    title="HealthText Cloud & Smart Record + MediChat API V5",
    description="Backend híbrido (Supabase + MongoDB HealthTextV2) con CNN, NLP y Búsqueda Heurística.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(healthtext_v2_router)

setup_storage()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.on_event("startup")
def startup_sync():
    """Sincroniza esquema de Supabase, cuenta demo de Paciente y médicos activos en MongoDB HealthTextV2."""
    try:
        ensure_v2_schema_and_sync()
        print("✅ Esquema V2, cuenta demo Paciente y doctores reales sincronizados en HealthTextV2.")
    except Exception as e:
        print(f"⚠️ Advertencia durante sincronización inicial: {e}")


def _get_live_mongo_doctors() -> list:
    if doctors_col is None:
        return []
    try:
        return list(doctors_col.find({"disponible": {"$ne": False}}, {"_id": 0}))
    except Exception as mongo_err:
        print(f"Advertencia: No se pudieron obtener los doctores de HealthTextV2: {mongo_err}")
        return []


class ChatInput(BaseModel):
    user_id: Optional[str] = "anonimo"
    message: str = Field(validation_alias=AliasChoices("message", "mensaje"))
    medical_context: Optional[str] = Field(
        default="Consulta Médica General",
        validation_alias=AliasChoices("medical_context", "contexto_medico"),
    )
    ai_grade: Optional[int] = Field(default=None, validation_alias=AliasChoices("ai_grade", "grado_ia"))


class ChatOutput(BaseModel):
    response: Dict[str, Any] = Field(default_factory=dict)
    respuesta: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context: Any) -> None:
        if self.respuesta is None and self.response:
            self.respuesta = self.response
        elif not self.response and self.respuesta:
            self.response = self.respuesta


class ClinicalNoteInput(BaseModel):
    symptoms: str = ""
    notes: str = ""
    diagnosis: str = ""


class AppointmentOptimizeRequest(BaseModel):
    patient_id: str
    patient_name: str
    risk: str = "Medio"
    specialty: str = "Medicina General"
    duration: int = 30
    preferred_date: Optional[str] = None
    reason: str = "Consulta de valoración clínica"


@app.post("/chat", response_model=ChatOutput)
async def handle_chat(chat_input: ChatInput):
    try:
        doctors_list = _get_live_mongo_doctors()
        parsed_response = await get_gemini_response(
            user_message=chat_input.message,
            medical_context=chat_input.medical_context or "Consulta Médica General",
            mongo_doctors_text=str(doctors_list),
            ai_grade=chat_input.ai_grade,
        )
        return ChatOutput(response=parsed_response)
    except Exception as e:
        print(f"Error en endpoint /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-burn")
async def analyze_burn(file: UploadFile = File(...)):
    try:
        return await process_burn_logic(file)
    except Exception as e:
        print(f"Error en analyze-burn: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-full")
async def analyze_full(
    file: Optional[UploadFile] = File(None),
    user_text: str = Form(...),
):
    try:
        doctors_list = _get_live_mongo_doctors()
        if file is None:
            ai_response = await get_gemini_response(
                user_message=user_text,
                medical_context="Consulta Médica General",
                mongo_doctors_text=str(doctors_list),
            )
            return {
                "visual_analysis": {"detail": "No se proporcionó imagen", "detalle": "No se proporcionó imagen"},
                "ai_diagnosis": ai_response,
                "user_text": user_text,
                "analisis_visual": {"detalle": "No se proporcionó imagen"},
                "diagnostico_ia": ai_response,
                "texto_usuario": user_text,
            }
        return await handle_integration_logic(file, user_text, doctors_list)
    except Exception as e:
        print(f"Error detallado en analyze-full: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/nlp/analyze-note")
def analyze_note_endpoint(payload: ClinicalNoteInput):
    return analyze_clinical_note(
        symptoms=payload.symptoms,
        notes=payload.notes,
        diagnosis=payload.diagnosis,
    )


@app.get("/api/nlp/metrics")
def get_nlp_metrics_endpoint():
    return get_nlp_evaluation_metrics()


@app.post("/api/appointments/optimize")
def optimize_appointment_endpoint(payload: AppointmentOptimizeRequest):
    result = optimize_appointment_schedule(
        patient_id=payload.patient_id,
        patient_name=payload.patient_name,
        risk=payload.risk,
        specialty=payload.specialty,
        duration=payload.duration,
        preferred_date=payload.preferred_date,
    )
    if result["recommended_slot"]:
        result["recommended_slot"]["reason"] = payload.reason
    return result


@app.get("/")
def read_root():
    return {
        "status": "Online",
        "model": "CNN-V5 + NLP + Heuristic Ready",
        "database": "Hybrid Supabase + MongoDB HealthTextV2",
        "api_version": "2.1.0",
    }


@app.get("/api/health/databases")
def database_health_endpoint():
    return check_hybrid_health()