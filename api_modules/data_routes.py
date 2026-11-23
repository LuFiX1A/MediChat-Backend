"""Endpoints de datos (lectura y autenticación) para HealthText.

Se registran en main.py con app.include_router(data_router).
No modifica los endpoints del chatbot (/chat, /analyze-burn, /analyze-full).
"""
from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db import MONGO_DATABASE, mongo_client, mongo_db, pg_all, pg_one, pg_connection

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
    # Nunca devolver password_hash al frontend.
    try:
        return pg_all(
            "SELECT id, name, email, role, specialty, status, last_access, color "
            "FROM users ORDER BY id"
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error consultando Supabase: {e}")


# --------------------------- AUTENTICACIÓN ---------------------------------

class LoginIn(BaseModel):
    email: str
    password: str


class RegisterIn(BaseModel):
    name: str
    email: str
    password: str
    role: Literal["Administrador", "Médico", "Enfermería", "Auditor"]
    color: str = "#2f9cc9"


@router.get("/users/exists")
def correo_existe(email: str):
    """Usado por el formulario de registro para validar correo duplicado."""
    try:
        fila = pg_one(
            "SELECT 1 AS ok FROM users WHERE lower(email) = %s",
            (email.strip().lower(),),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error consultando Supabase: {e}")
    return {"exists": fila is not None}


@router.post("/login")
def login(body: LoginIn):
    """Valida credenciales contra users de Supabase. No devuelve password_hash."""
    try:
        fila = pg_one(
            "SELECT id, name, email, role, specialty, password_hash "
            "FROM users WHERE lower(email) = %s",
            (body.email.strip().lower(),),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error consultando Supabase: {e}")

    # El seed usa contraseñas en texto plano solo con fines didácticos.
    if fila is None or fila["password_hash"] != body.password:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    # Actualizar último acceso no debe impedir el login si falla.
    try:
        with pg_connection() as conn:
            conn.execute(
                "UPDATE users SET last_access = %s WHERE id = %s",
                ("Hoy, " + datetime.now().strftime("%H:%M"), fila["id"]),
            )
            conn.commit()
    except Exception:
        pass

    return {k: fila[k] for k in ("id", "name", "email", "role", "specialty")}


@router.post("/register", status_code=201)
def registrar(body: RegisterIn):
    """Crea la cuenta directamente en users de Supabase."""
    email = body.email.strip().lower()
    nombre = body.name.strip()

    try:
        existe = pg_one("SELECT 1 AS ok FROM users WHERE lower(email) = %s", (email,))
        if existe is not None:
            raise HTTPException(status_code=409, detail="Ya existe una cuenta con ese correo")

        nuevo_id = f"u-reg-{uuid4().hex[:8]}"
        with pg_connection() as conn:
            conn.execute(
                "INSERT INTO users (id, name, email, password_hash, role, status, "
                "last_access, color) VALUES (%s, %s, %s, %s, %s, 'Activo', "
                "'Registro reciente', %s)",
                (nuevo_id, nombre, email, body.password, body.role, body.color),
            )
            conn.commit()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error registrando en Supabase: {e}")

    return {"id": nuevo_id, "name": nombre, "email": email, "role": body.role}


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
