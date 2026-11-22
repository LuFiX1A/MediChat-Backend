"""Endpoints de solo lectura para las bases de datos de HealthText.

Se registran en main.py con app.include_router(data_router).
No modifica los endpoints del chatbot (/chat, /analyze-burn, /analyze-full).
"""
from typing import Optional

from fastapi import APIRouter, HTTPException

from db import MONGO_DATABASE, mongo_client, mongo_db, pg_all, pg_one

router = APIRouter()


@router.get("/db-status")
def db_status():
    """Diagnóstico rápido de ambas conexiones."""
    supabase_ok = False
    try:
        pg_one("SELECT 1 AS ok")
        supabase_ok = True
    except Exception:
        pass

    mongo_ok = False
    try:
        if mongo_client is not None:
            mongo_client.admin.command("ping")
            mongo_ok = True
    except Exception:
        pass

    return {
        "supabase": supabase_ok,
        "mongodb": mongo_ok,
        "mongo_database": MONGO_DATABASE,
    }


# ------------------------- SUPABASE (PostgreSQL) -------------------------

@router.get("/patients")
def listar_pacientes():
    try:
        return pg_all("SELECT * FROM patients ORDER BY id")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error consultando Supabase: {e}")


@router.get("/patients/{patient_id}")
def obtener_paciente(patient_id: str):
    try:
        fila = pg_one("SELECT * FROM patients WHERE id = %s", (patient_id,))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error consultando Supabase: {e}")
    if fila is None:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    return fila


@router.get("/patients/{patient_id}/vitals")
def signos_vitales(patient_id: str):
    try:
        return pg_all(
            "SELECT * FROM patient_vitals WHERE patient_id = %s ORDER BY recorded_at DESC",
            (patient_id,),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error consultando Supabase: {e}")


@router.get("/appointments")
def listar_citas():
    try:
        return pg_all("SELECT * FROM appointments ORDER BY date, time")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error consultando Supabase: {e}")


@router.get("/users")
def listar_usuarios():
    try:
        return pg_all("SELECT * FROM users ORDER BY id")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error consultando Supabase: {e}")


@router.get("/audit-logs")
def bitacora():
    try:
        return pg_all("SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT 200")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error consultando Supabase: {e}")


@router.get("/honeypot-events")
def eventos_honeypot():
    try:
        return pg_all("SELECT * FROM honeypot_events ORDER BY timestamp DESC LIMIT 200")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error consultando Supabase: {e}")


# --------------------------- MONGODB ATLAS --------------------------------

@router.get("/medical-records/{patient_id}")
def expediente_medico(patient_id: str):
    if mongo_db is None:
        raise HTTPException(status_code=503, detail="MONGO_URI no está configurada")
    try:
        doc = mongo_db["medical_records"].find_one(
            {"patient_id": patient_id}, {"_id": 0}
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error consultando MongoDB: {e}")
    if doc is None:
        raise HTTPException(status_code=404, detail="Expediente no encontrado")
    return doc


@router.get("/triage-sessions")
def sesiones_triaje(patient_id: Optional[str] = None):
    if mongo_db is None:
        raise HTTPException(status_code=503, detail="MONGO_URI no está configurada")
    filtro = {"patient_id": patient_id} if patient_id else {}
    try:
        return list(mongo_db["triage_sessions"].find(filtro, {"_id": 0}))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error consultando MongoDB: {e}")
