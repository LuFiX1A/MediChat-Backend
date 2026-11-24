"""Endpoints de datos (lectura y autenticación) para HealthText.

Se registran en main.py con app.include_router(data_router).
No modifica los endpoints del chatbot (/chat, /analyze-burn, /analyze-full).
"""
from datetime import date, datetime
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


# --------------------------- ESCRITURA -------------------------------------

class VitalsIn(BaseModel):
    pressure: str
    heart_rate: int
    temperature: float
    weight: float
    height: float


class PatientIn(BaseModel):
    id: str
    name: str
    age: int
    sex: str
    document: str
    birth_date: date
    blood_type: str
    phone: str
    email: str
    address: str
    insurance: str
    last_visit: date
    status: str
    risk: str
    color: str = "#2f9cc9"
    allergies: list[str] = []
    conditions: list[str] = []
    medications: list[str] = []
    vitals: VitalsIn


class AppointmentIn(BaseModel):
    id: str
    patient_id: str
    patient_name: str
    doctor_id: str | None = None
    doctor: str
    specialty: str
    date: date
    time: str
    duration: int = 30
    reason: str
    status: str
    priority_score: int = 1


class ConsultationIn(BaseModel):
    id: str
    date: date
    doctor: str
    specialty: str
    reason: str
    symptoms: str
    notes: str
    diagnosis: str
    treatment: str
    status: str
    risk: str


class UserIn(BaseModel):
    id: str
    name: str
    email: str
    role: str
    status: str = "Activo"
    color: str = "#2f9cc9"
    password_hash: str = "demo1234"


class StatusIn(BaseModel):
    status: str


class AuditIn(BaseModel):
    id: str
    date: date
    time: str
    user_name: str
    action: str
    module: str
    detail: str
    ip: str
    severity: str


class HoneypotIn(BaseModel):
    id: str
    timestamp: datetime
    type: str
    severity: str
    route: str
    detail: str
    user_name: str
    role: str
    origin: str
    user_agent: str | None = None
    attempted_user: str | None = None


@router.post("/patients", status_code=201)
def crear_paciente(body: PatientIn):
    with pg_connection() as conn:
        conn.execute(
            """INSERT INTO patients
               (id, name, age, sex, document, birth_date, blood_type, phone, email,
                address, insurance, last_visit, status, risk, color)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (body.id, body.name, body.age, body.sex, body.document, body.birth_date,
             body.blood_type, body.phone, body.email, body.address, body.insurance,
             body.last_visit, body.status, body.risk, body.color),
        )
        conn.execute(
            """INSERT INTO patient_vitals
               (patient_id, pressure, heart_rate, temperature, weight, height)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (body.id, body.vitals.pressure, body.vitals.heart_rate,
             body.vitals.temperature, body.vitals.weight, body.vitals.height),
        )
        conn.commit()

    if mongo_db is not None:
        mongo_db["medical_records"].update_one(
            {"patient_id": body.id},
            {"$set": {
                "allergies": body.allergies,
                "conditions": body.conditions,
                "medications": body.medications,
            }, "$setOnInsert": {"consultations": []}},
            upsert=True,
        )
    return {"id": body.id}


@router.patch("/patients/{patient_id}/vitals")
def actualizar_vitals(patient_id: str, body: VitalsIn):
    with pg_connection() as conn:
        fila = conn.execute(
            """INSERT INTO patient_vitals
               (patient_id, pressure, heart_rate, temperature, weight, height)
               VALUES (%s,%s,%s,%s,%s,%s) RETURNING *""",
            (patient_id, body.pressure, body.heart_rate,
             body.temperature, body.weight, body.height),
        ).fetchone()
        conn.commit()
    return fila


@router.post("/consultations/{patient_id}", status_code=201)
def agregar_consulta(patient_id: str, body: ConsultationIn):
    if mongo_db is None:
        raise HTTPException(status_code=503, detail="MONGO_URI no está configurada")
    datos = body.model_dump()
    consulta = {
        "id": body.id,
        **{k: (v.isoformat() if isinstance(v, date) else v) for k, v in datos.items()},
    }
    mongo_db["medical_records"].update_one(
        {"patient_id": patient_id},
        {"$push": {"consultations": consulta}},
        upsert=True,
    )
    with pg_connection() as conn:
        conn.execute(
            "UPDATE patients SET last_visit = %s, status = 'En seguimiento' WHERE id = %s",
            (body.date, patient_id),
        )
        conn.commit()
    return consulta


@router.post("/appointments", status_code=201)
def crear_cita(body: AppointmentIn):
    with pg_connection() as conn:
        conn.execute(
            """INSERT INTO appointments
               (id, patient_id, patient_name, doctor_id, doctor, specialty, date,
                time, duration, reason, status, priority_score)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (body.id, body.patient_id, body.patient_name, body.doctor_id, body.doctor,
             body.specialty, body.date, body.time, body.duration, body.reason,
             body.status, body.priority_score),
        )
        conn.commit()
    return {"id": body.id}


@router.patch("/appointments/{appointment_id}")
def actualizar_cita(appointment_id: str, body: AppointmentIn):
    with pg_connection() as conn:
        conn.execute(
            """UPDATE appointments SET patient_id=%s, patient_name=%s, doctor_id=%s,
               doctor=%s, specialty=%s, date=%s, time=%s, duration=%s, reason=%s,
               status=%s, priority_score=%s WHERE id=%s""",
            (body.patient_id, body.patient_name, body.doctor_id, body.doctor,
             body.specialty, body.date, body.time, body.duration, body.reason,
             body.status, body.priority_score, appointment_id),
        )
        conn.commit()
    return {"id": appointment_id}


@router.post("/users", status_code=201)
def crear_usuario(body: UserIn):
    with pg_connection() as conn:
        conn.execute(
            """INSERT INTO users (id, name, email, password_hash, role, status,
               last_access, color) VALUES (%s,%s,%s,%s,%s,%s,'Sin acceso',%s)""",
            (body.id, body.name, body.email, body.password_hash, body.role,
             body.status, body.color),
        )
        conn.commit()
    return {"id": body.id}


@router.patch("/users/{user_id}/status")
def cambiar_estado_usuario(user_id: str, body: StatusIn):
    with pg_connection() as conn:
        fila = conn.execute(
            "UPDATE users SET status = %s WHERE id = %s RETURNING id, status",
            (body.status, user_id),
        ).fetchone()
        conn.commit()
    if fila is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return fila


@router.post("/audit-logs", status_code=201)
def registrar_auditoria(body: AuditIn):
    with pg_connection() as conn:
        conn.execute(
            """INSERT INTO audit_logs
               (id, date, time, user_name, action, module, detail, ip, severity)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (body.id, body.date, body.time, body.user_name, body.action,
             body.module, body.detail, body.ip, body.severity),
        )
        conn.commit()
    return {"id": body.id}


@router.post("/honeypot-events", status_code=201)
def registrar_honeypot(body: HoneypotIn):
    with pg_connection() as conn:
        conn.execute(
            """INSERT INTO honeypot_events
               (id, timestamp, type, severity, route, detail, user_name, role,
                origin, user_agent, attempted_user)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (body.id, body.timestamp, body.type, body.severity, body.route,
             body.detail, body.user_name, body.role, body.origin,
             body.user_agent, body.attempted_user),
        )
        conn.commit()
    return {"id": body.id}


@router.post("/triage-sessions", status_code=201)
def crear_triaje(body: dict):
    if mongo_db is None:
        raise HTTPException(status_code=503, detail="MONGO_URI no está configurada")
    mongo_db["triage_sessions"].insert_one(body)
    return {"ok": True}


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
