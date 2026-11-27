import uuid
from datetime import datetime, date
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from database.hybrid_db import (
    get_pg_connection,
    doctors_col,
    records_col,
    triage_col,
)
from core.security import (
    verify_password,
    hash_password,
    create_access_token,
)
from services.nlp_service import analyze_clinical_note
from services.scheduler_service import optimize_appointment_schedule

router = APIRouter(prefix="/api", tags=["HealthText Cloud & Smart Record V2"])


# ==============================================================================
# SINCRONIZACIÓN DE ESQUEMA, PACIENTE DEMO Y DOCTORES REALES
# ==============================================================================
def ensure_v2_schema_and_sync() -> None:
    """Garantiza columnas de validación médica, paciente demo y doctores reales en MongoDB."""
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS cedula VARCHAR(60);")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS credential VARCHAR(80);")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS patient_id VARCHAR(50);")
            cur.execute("ALTER TABLE appointments ADD COLUMN IF NOT EXISTS origin VARCHAR(40) DEFAULT 'Clínica';")
            cur.execute("ALTER TABLE appointments ADD COLUMN IF NOT EXISTS urgent BOOLEAN DEFAULT FALSE;")

            # Asegurar cuenta demo de Paciente vinculada a María Fernanda López (p-001)
            cur.execute("SELECT id FROM patients ORDER BY created_at ASC LIMIT 1;")
            first_patient = cur.fetchone()
            linked_pid = first_patient["id"] if first_patient else "p-001"
            linked_name = "María Fernanda López"
            if first_patient:
                cur.execute("SELECT name FROM patients WHERE id = %s;", (linked_pid,))
                p_row = cur.fetchone()
                if p_row:
                    linked_name = p_row["name"]

            cur.execute("SELECT id FROM users WHERE LOWER(email) = 'paciente@healthtext.com';")
            if not cur.fetchone():
                cur.execute(
                    """
                    INSERT INTO users (id, name, email, password_hash, role, specialty, status, last_access, color, patient_id)
                    VALUES ('u-pac-demo', %s, 'paciente@healthtext.com', %s, 'Paciente', NULL, 'Activo', 'Cuenta demo', '#0ea5e9', %s)
                    """,
                    (linked_name, hash_password("demo1234"), linked_pid),
                )

            cur.execute(
                """
                SELECT id, name, COALESCE(specialty, 'Medicina General') AS specialty,
                       status, COALESCE(cedula, '') AS cedula, COALESCE(credential, '') AS credential
                FROM users
                WHERE role = 'Médico'
                ORDER BY created_at ASC
                """
            )
            pg_doctors = cur.fetchall()
        conn.commit()

    # Sincronizar exactamente los doctores reales de Supabase hacia MongoDB HealthTextV2.doctors
    if doctors_col is not None and pg_doctors:
        valid_ids = [d["id"] for d in pg_doctors]
        doctors_col.delete_many({"mongo_id": {"$nin": valid_ids}})
        for doc in pg_doctors:
            doctors_col.update_one(
                {"mongo_id": doc["id"]},
                {
                    "$set": {
                        "mongo_id": doc["id"],
                        "nombre_doctor": doc["name"],
                        "especialidad": doc["specialty"] or "Medicina General",
                        "cedula": doc["cedula"],
                        "credencial": doc["credential"],
                        "subespecialidades": ["Atención Clínica Integral", doc["specialty"] or "Medicina General"],
                        "disponible": doc["status"] == "Activo",
                    }
                },
                upsert=True,
            )


# ==============================================================================
# MODELOS PYDANTIC
# ==============================================================================
class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str = "Paciente"
    document: Optional[str] = None
    age: int = 28
    sex: str = "Femenino"
    phone: str = "+52 449 000 0000"
    insurance: str = "IMSS Bienestar"
    specialty: Optional[str] = None


class UserCreateRequest(BaseModel):
    name: str
    email: str
    role: str
    specialty: Optional[str] = None
    cedula: Optional[str] = None
    credential: Optional[str] = None
    password: str = "demo1234"


class VitalSignsInput(BaseModel):
    pressure: str = "120/80"
    heartRate: int = 75
    temperature: float = 36.6
    weight: float = 70.0
    height: float = 1.70


class PatientCreateRequest(BaseModel):
    name: str
    age: int
    sex: str
    document: str
    birthDate: str
    bloodType: str
    phone: str
    email: str
    address: str
    insurance: str
    allergies: List[str] = []
    conditions: List[str] = []
    medications: List[str] = []
    status: str = "Activo"
    risk: str = "Bajo"
    color: str = "#2f9cc9"
    vitals: VitalSignsInput


class ConsultationCreateRequest(BaseModel):
    date: str
    doctor: str
    specialty: str
    reason: str
    symptoms: str
    notes: str
    diagnosis: str
    treatment: str
    status: str = "Seguimiento"
    risk: Optional[str] = None


class TriageSaveRequest(BaseModel):
    summary: str
    recommended_doctor: str = ""
    recommended_specialty: str = ""
    burn_grade: Optional[int] = None


class AppointmentCreateRequest(BaseModel):
    patientId: str
    patientName: str
    doctorId: Optional[str] = None
    doctor: str
    specialty: str
    date: str
    time: str
    duration: int = 30
    reason: str
    status: str = "Confirmada"
    priorityScore: int = 1
    origin: str = "Clínica"
    urgent: bool = False


class AppointmentStatusUpdate(BaseModel):
    status: str


class AppointmentOptimizePayload(BaseModel):
    patient_id: str
    patient_name: str
    risk: str = "Medio"
    specialty: str = "Medicina General"
    duration: int = 30
    preferred_date: Optional[str] = None
    reason: str = "Valoración clínica"


class AuditCreateRequest(BaseModel):
    user: str
    action: str
    module: str
    detail: str
    severity: str = "info"


class HoneypotCreateRequest(BaseModel):
    type: str
    severity: str = "critical"
    route: str
    detail: str
    user: str
    role: str
    origin: str = "203.0.113.10"
    userAgent: str = "Unknown"
    attemptedUser: Optional[str] = None


# ==============================================================================
# UTILIDADES INTERNAS
# ==============================================================================
def _initials(name: str) -> str:
    parts = [p for p in name.strip().split() if p]
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return name[:2].upper() if name else "US"


def _write_audit_pg(
    user_name: str,
    action: str,
    module: str,
    detail: str,
    severity: str = "info",
    ip: str = "127.0.0.1",
) -> dict:
    now = datetime.now()
    log_id = f"log-{uuid.uuid4().hex[:8]}"
    date_str = now.date().isoformat()
    time_str = now.strftime("%H:%M:%S")
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_logs (id, date, time, user_name, action, module, detail, ip, severity)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (log_id, date_str, time_str, user_name, action, module, detail, ip, severity),
            )
        conn.commit()
    return {
        "id": log_id,
        "date": date_str,
        "time": time_str,
        "user": user_name,
        "action": action,
        "module": module,
        "detail": detail,
        "ip": ip,
        "severity": severity,
    }


# ==============================================================================
# 1. AUTENTICACIÓN Y REGISTRO PÚBLICO DE PACIENTES
# ==============================================================================
@router.post("/auth/login")
def login(payload: LoginRequest, request: Request):
    email_norm = payload.email.strip().lower()
    client_ip = request.client.host if request.client else "127.0.0.1"

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE LOWER(email) = %s", (email_norm,))
            user = cur.fetchone()

            if not user or not verify_password(payload.password, user["password_hash"]):
                _write_audit_pg(
                    email_norm,
                    "Intento de inicio de sesión fallido",
                    "Autenticación",
                    f"Credenciales inválidas para {email_norm}",
                    "warning",
                    client_ip,
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Correo o contraseña incorrectos",
                )

            if user["status"] != "Activo":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Usuario inactivo. Contacte al administrador.",
                )

            cur.execute(
                "UPDATE users SET last_access = %s WHERE id = %s",
                (datetime.now().strftime("Hoy, %H:%M"), user["id"]),
            )
        conn.commit()

    token = create_access_token(
        {
            "sub": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
            "patientId": user.get("patient_id"),
        }
    )
    _write_audit_pg(
        user["name"],
        "Inicio de sesión exitoso",
        "Autenticación",
        f"Acceso autorizado con rol {user['role']}",
        "info",
        client_ip,
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "initials": _initials(user["name"]),
            "patientId": user.get("patient_id"),
        },
    }


@router.post("/auth/register")
def register(payload: RegisterRequest, request: Request):
    """Autorregistro público exclusivo para Pacientes (crea cuenta + expediente clínico)."""
    email_norm = payload.email.strip().lower()
    client_ip = request.client.host if request.client else "127.0.0.1"
    user_id = f"u-{uuid.uuid4().hex[:6]}"
    patient_id = f"p-{uuid.uuid4().hex[:6]}"
    hashed = hash_password(payload.password)
    today_str = date.today().isoformat()
    doc_code = (payload.document or f"CURP-{uuid.uuid4().hex[:8].upper()}").strip()

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE LOWER(email) = %s", (email_norm,))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="El correo ya está registrado")

            cur.execute(
                """
                INSERT INTO patients (
                    id, name, age, sex, document, birth_date, blood_type,
                    phone, email, address, insurance, last_visit, status, risk, color, is_honeytoken
                ) VALUES (%s, %s, %s, %s, %s, '1998-01-01', 'O+', %s, %s, 'Registrado desde Portal Paciente', %s, %s, 'Activo', 'Bajo', '#0ea5e9', FALSE)
                """,
                (
                    patient_id,
                    payload.name.strip(),
                    payload.age,
                    payload.sex,
                    doc_code,
                    payload.phone,
                    email_norm,
                    payload.insurance,
                    today_str,
                ),
            )
            cur.execute(
                """
                INSERT INTO patient_vitals (patient_id, pressure, heart_rate, temperature, weight, height)
                VALUES (%s, '120/80', 74, 36.6, 68.0, 1.68)
                """,
                (patient_id,),
            )
            cur.execute(
                """
                INSERT INTO users (id, name, email, password_hash, role, specialty, status, last_access, color, patient_id)
                VALUES (%s, %s, %s, %s, 'Paciente', NULL, 'Activo', 'Recién registrado', '#0ea5e9', %s)
                """,
                (user_id, payload.name.strip(), email_norm, hashed, patient_id),
            )
        conn.commit()

    if records_col is not None:
        records_col.update_one(
            {"patient_id": patient_id},
            {
                "$set": {
                    "patient_id": patient_id,
                    "allergies": [],
                    "conditions": [],
                    "medications": [],
                    "consultations": [],
                    "triage_notes": [],
                }
            },
            upsert=True,
        )

    _write_audit_pg(
        payload.name.strip(),
        "Registro de nuevo paciente",
        "Portal Paciente",
        f"Cuenta y expediente {patient_id} creados desde registro público",
        "info",
        client_ip,
    )

    return {
        "id": user_id,
        "name": payload.name.strip(),
        "email": email_norm,
        "role": "Paciente",
        "initials": _initials(payload.name),
        "patientId": patient_id,
    }


# ==============================================================================
# 2. GESTIÓN DE USUARIOS Y DOCTORES (SUPABASE + MONGODB SYNC)
# ==============================================================================
@router.get("/users")
def list_users():
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, email, role, specialty,
                       COALESCE(cedula, '') AS cedula,
                       COALESCE(credential, '') AS credential,
                       status, last_access AS "lastAccess", color,
                       patient_id AS "patientId"
                FROM users
                ORDER BY created_at ASC
                """
            )
            return cur.fetchall()


@router.post("/users")
def create_system_user(payload: UserCreateRequest):
    user_id = f"u-{uuid.uuid4().hex[:6]}"
    email_norm = payload.email.strip().lower()
    hashed = hash_password(payload.password)

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE LOWER(email) = %s", (email_norm,))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="Correo duplicado")
            cur.execute(
                """
                INSERT INTO users (id, name, email, password_hash, role, specialty, cedula, credential, status, last_access, color)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'Activo', 'Nunca', '#5b8def')
                """,
                (
                    user_id,
                    payload.name.strip(),
                    email_norm,
                    hashed,
                    payload.role,
                    payload.specialty,
                    payload.cedula,
                    payload.credential,
                ),
            )
        conn.commit()

    if payload.role == "Médico" and doctors_col is not None:
        doctors_col.update_one(
            {"mongo_id": user_id},
            {
                "$set": {
                    "mongo_id": user_id,
                    "nombre_doctor": payload.name.strip(),
                    "especialidad": payload.specialty or "Medicina General",
                    "cedula": payload.cedula or "",
                    "credencial": payload.credential or "",
                    "subespecialidades": ["Atención Clínica Integral", payload.specialty or "Medicina General"],
                    "disponible": True,
                }
            },
            upsert=True,
        )

    _write_audit_pg(
        "Administrador",
        "Usuario creado",
        "Administración",
        f"Alta de {payload.name} ({payload.role}) · Cédula: {payload.cedula or 'N/A'}",
        "info",
    )
    return {
        "id": user_id,
        "name": payload.name.strip(),
        "email": email_norm,
        "role": payload.role,
        "specialty": payload.specialty,
        "cedula": payload.cedula or "",
        "credential": payload.credential or "",
        "status": "Activo",
        "lastAccess": "Nunca",
        "color": "#5b8def",
    }


@router.patch("/users/{user_id}/toggle-status")
def toggle_user_status(user_id: str):
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, status FROM users WHERE id = %s", (user_id,))
            user = cur.fetchone()
            if not user:
                raise HTTPException(status_code=404, detail="Usuario no encontrado")
            new_status = "Inactivo" if user["status"] == "Activo" else "Activo"
            cur.execute("UPDATE users SET status = %s WHERE id = %s", (new_status, user_id))
        conn.commit()

    if doctors_col is not None:
        doctors_col.update_one({"mongo_id": user_id}, {"$set": {"disponible": new_status == "Activo"}})

    _write_audit_pg("Administrador", "Cambio de estado de usuario", "Administración", f"{user['name']} -> {new_status}", "warning")
    return {"id": user_id, "status": new_status}


# ==============================================================================
# 3. EXPEDIENTE MÉDICO ELECTRÓNICO HÍBRIDO (PACIENTES + CONSULTAS + TRIAJE)
# ==============================================================================
def _hydrate_patient(row: dict) -> dict:
    pid = row["id"]
    doc_record = records_col.find_one({"patient_id": pid}, {"_id": 0}) if records_col is not None else None

    return {
        "id": pid,
        "name": row["name"],
        "age": row["age"],
        "sex": row["sex"],
        "document": row["document"],
        "birthDate": str(row["birth_date"]),
        "bloodType": row["blood_type"],
        "phone": row["phone"],
        "email": row["email"],
        "address": row["address"],
        "insurance": row["insurance"],
        "lastVisit": str(row["last_visit"]),
        "status": row["status"],
        "risk": row["risk"],
        "color": row["color"],
        "isHoneytoken": bool(row.get("is_honeytoken", False)),
        "vitals": {
            "pressure": row.get("pressure") or "120/80",
            "heartRate": row.get("heart_rate") or 75,
            "temperature": float(row.get("temperature") or 36.6),
            "weight": float(row.get("weight") or 70.0),
            "height": float(row.get("height") or 1.70),
        },
        "allergies": doc_record.get("allergies", []) if doc_record else [],
        "conditions": doc_record.get("conditions", []) if doc_record else [],
        "medications": doc_record.get("medications", []) if doc_record else [],
        "consultations": doc_record.get("consultations", []) if doc_record else [],
        "triageNotes": doc_record.get("triage_notes", []) if doc_record else [],
    }


@router.get("/patients")
def get_all_patients():
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.*, v.pressure, v.heart_rate, v.temperature, v.weight, v.height
                FROM patients p
                LEFT JOIN LATERAL (
                    SELECT pressure, heart_rate, temperature, weight, height
                    FROM patient_vitals
                    WHERE patient_id = p.id
                    ORDER BY recorded_at DESC
                    LIMIT 1
                ) v ON TRUE
                ORDER BY p.created_at ASC
                """
            )
            rows = cur.fetchall()
    return [_hydrate_patient(r) for r in rows]


@router.get("/patients/{patient_id}")
def get_patient_detail(patient_id: str, request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.*, v.pressure, v.heart_rate, v.temperature, v.weight, v.height
                FROM patients p
                LEFT JOIN LATERAL (
                    SELECT pressure, heart_rate, temperature, weight, height
                    FROM patient_vitals
                    WHERE patient_id = p.id
                    ORDER BY recorded_at DESC
                    LIMIT 1
                ) v ON TRUE
                WHERE p.id = %s
                """,
                (patient_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")

    if row.get("is_honeytoken"):
        hp_id = f"hp-{uuid.uuid4().hex[:8]}"
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO honeypot_events (id, type, severity, route, detail, user_name, role, origin, user_agent)
                    VALUES (%s, 'Honeytoken clínico', 'critical', %s, %s, 'Usuario en sesión', 'Clínico', %s, %s)
                    """,
                    (
                        hp_id,
                        f"/pacientes/{patient_id}",
                        f"Acceso detectado al expediente señuelo {row['name']} ({patient_id})",
                        client_ip,
                        request.headers.get("user-agent", "Desconocido"),
                    ),
                )
            conn.commit()
        _write_audit_pg(
            "Sistema de Seguridad",
            "Intruso detectado en honeypot",
            "Seguridad",
            f"Consulta de expediente señuelo {patient_id}",
            "critical",
            client_ip,
        )

    return _hydrate_patient(row)


@router.post("/patients")
def create_patient(payload: PatientCreateRequest):
    patient_id = f"p-{uuid.uuid4().hex[:6]}"
    today_str = date.today().isoformat()

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patients (
                    id, name, age, sex, document, birth_date, blood_type,
                    phone, email, address, insurance, last_visit, status, risk, color, is_honeytoken
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE)
                """,
                (
                    patient_id,
                    payload.name,
                    payload.age,
                    payload.sex,
                    payload.document,
                    payload.birthDate,
                    payload.bloodType,
                    payload.phone,
                    payload.email,
                    payload.address,
                    payload.insurance,
                    today_str,
                    payload.status,
                    payload.risk,
                    payload.color,
                ),
            )
            cur.execute(
                """
                INSERT INTO patient_vitals (patient_id, pressure, heart_rate, temperature, weight, height)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    patient_id,
                    payload.vitals.pressure,
                    payload.vitals.heartRate,
                    payload.vitals.temperature,
                    payload.vitals.weight,
                    payload.vitals.height,
                ),
            )
        conn.commit()

    if records_col is not None:
        records_col.update_one(
            {"patient_id": patient_id},
            {
                "$set": {
                    "patient_id": patient_id,
                    "allergies": payload.allergies,
                    "conditions": payload.conditions,
                    "medications": payload.medications,
                    "consultations": [],
                    "triage_notes": [],
                }
            },
            upsert=True,
        )

    _write_audit_pg("Personal Clínico", "Alta de paciente", "Pacientes", f"Nuevo expediente creado: {payload.name} ({patient_id})", "info")
    return {"id": patient_id, "name": payload.name, "status": "created"}


@router.post("/patients/{patient_id}/triage")
def save_patient_triage_from_medichat(patient_id: str, payload: TriageSaveRequest):
    """Guarda el resumen de triaje de MediChat en MongoDB y actualiza riesgo NLP del paciente."""
    nlp_result = analyze_clinical_note(symptoms=payload.summary, notes="", diagnosis=payload.recommended_specialty)
    detected_risk = nlp_result["predicted_risk"] if nlp_result["status"] == "ok" else "Medio"

    entry = {
        "id": f"tr-{uuid.uuid4().hex[:6]}",
        "date": date.today().isoformat(),
        "time": datetime.now().strftime("%H:%M"),
        "summary": payload.summary,
        "recommendedDoctor": payload.recommended_doctor,
        "recommendedSpecialty": payload.recommended_specialty,
        "burnGrade": payload.burn_grade,
        "nlpRisk": detected_risk,
    }

    if records_col is not None:
        records_col.update_one(
            {"patient_id": patient_id},
            {"$push": {"triage_notes": {"$each": [entry], "$position": 0}}},
            upsert=True,
        )

    if triage_col is not None:
        triage_col.insert_one({"patient_id": patient_id, **entry})

    if detected_risk == "Alto":
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE patients SET risk = 'Alto' WHERE id = %s", (patient_id,))
            conn.commit()

    return entry


@router.post("/patients/{patient_id}/consultations")
def add_patient_consultation(patient_id: str, payload: ConsultationCreateRequest):
    nlp_result = analyze_clinical_note(
        symptoms=payload.symptoms,
        notes=payload.notes,
        diagnosis=payload.diagnosis,
    )
    if nlp_result["status"] == "ok" and nlp_result["confidence"] > 0:
        final_risk = nlp_result["predicted_risk"]
    else:
        final_risk = payload.risk if payload.risk in ("Bajo", "Medio", "Alto") else "Medio"
    consultation_id = f"c-{uuid.uuid4().hex[:6]}"

    consultation_doc = {
        "id": consultation_id,
        "date": payload.date,
        "doctor": payload.doctor,
        "specialty": payload.specialty,
        "reason": payload.reason,
        "symptoms": payload.symptoms,
        "notes": payload.notes,
        "diagnosis": payload.diagnosis,
        "treatment": payload.treatment,
        "status": payload.status,
        "risk": final_risk,
        "nlp_analysis": {
            "predicted_risk": nlp_result["predicted_risk"],
            "confidence": nlp_result["confidence"],
            "extracted_keywords": nlp_result["extracted_keywords"],
        },
    }

    if records_col is not None:
        records_col.update_one(
            {"patient_id": patient_id},
            {"$push": {"consultations": {"$each": [consultation_doc], "$position": 0}}},
            upsert=True,
        )

    new_patient_status = "Activo" if payload.status == "Finalizada" else "En seguimiento"
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE patients
                SET last_visit = %s, risk = %s, status = %s
                WHERE id = %s
                """,
                (payload.date, final_risk, new_patient_status, patient_id),
            )
        conn.commit()

    _write_audit_pg(
        payload.doctor,
        "Registro de consulta clínica + Análisis NLP",
        "Consultas",
        f"Paciente {patient_id} | Riesgo clasificado: {final_risk} (confianza {nlp_result['confidence']})",
        "warning" if final_risk == "Alto" else "info",
    )

    return consultation_doc


# ==============================================================================
# 4. GESTIÓN DE CITAS Y SOLICITUDES DESDE MEDICHAT
# ==============================================================================
@router.get("/appointments")
def list_appointments():
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, patient_id AS "patientId", patient_name AS "patientName",
                       doctor_id AS "doctorId", doctor, specialty,
                       TO_CHAR(date, 'YYYY-MM-DD') AS date, time, duration, reason, status,
                       priority_score AS "priorityScore",
                       COALESCE(origin, 'Clínica') AS origin,
                       COALESCE(urgent, FALSE) AS urgent
                FROM appointments
                ORDER BY date ASC, time ASC
                """
            )
            return cur.fetchall()


@router.post("/appointments")
def create_appointment(payload: AppointmentCreateRequest):
    appt_id = f"a-{uuid.uuid4().hex[:6]}"
    priority = payload.priorityScore

    # Normalizar hora a HH:MM por compatibilidad con columnas VARCHAR(10) / TIME en Supabase
    raw_time = (payload.time or "").strip()
    if len(raw_time) > 8 or not any(ch.isdigit() for ch in raw_time):
        clean_time = "08:00" if payload.urgent else "09:00"
    else:
        clean_time = raw_time[:8]

    # Si proviene de MediChat, calcular prioridad automáticamente con NLP sobre el motivo/síntomas
    if payload.origin == "MediChat":
        nlp_res = analyze_clinical_note(symptoms=payload.reason, notes="", diagnosis=payload.specialty)
        if payload.urgent or nlp_res.get("predicted_risk") == "Alto":
            priority = 3
        elif nlp_res.get("predicted_risk") == "Medio":
            priority = 2

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            # Verificar que patient_id exista en patients para evitar violación de FK
            cur.execute("SELECT id FROM patients WHERE id = %s;", (payload.patientId,))
            valid_patient = cur.fetchone()
            target_patient_id = payload.patientId

            if not valid_patient:
                cur.execute(
                    "SELECT id FROM patients WHERE LOWER(name) = LOWER(%s) LIMIT 1;",
                    (payload.patientName,),
                )
                by_name = cur.fetchone()
                if by_name:
                    target_patient_id = by_name["id"]
                else:
                    cur.execute("SELECT id FROM patients ORDER BY created_at ASC LIMIT 1;")
                    fallback_p = cur.fetchone()
                    if fallback_p:
                        target_patient_id = fallback_p["id"]

            cur.execute(
                """
                INSERT INTO appointments (
                    id, patient_id, patient_name, doctor_id, doctor, specialty,
                    date, time, duration, reason, status, priority_score, origin, urgent
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    appt_id,
                    target_patient_id,
                    payload.patientName,
                    payload.doctorId,
                    payload.doctor,
                    payload.specialty,
                    payload.date,
                    clean_time,
                    payload.duration,
                    payload.reason,
                    payload.status,
                    priority,
                    payload.origin,
                    payload.urgent,
                ),
            )
        conn.commit()

    _write_audit_pg(
        payload.patientName if payload.origin == "MediChat" else "Agenda Clínica",
        "Solicitud de cita desde MediChat" if payload.origin == "MediChat" else "Cita agendada",
        "Citas",
        f"{payload.patientName} con {payload.doctor} ({payload.date} {clean_time})",
        "warning" if payload.urgent else "info",
    )
    return {
        "id": appt_id,
        **payload.model_dump(),
        "patientId": target_patient_id,
        "time": clean_time,
        "priorityScore": priority,
    }


@router.patch("/appointments/{appointment_id}/status")
def update_appointment_status(appointment_id: str, payload: AppointmentStatusUpdate):
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE appointments SET status = %s WHERE id = %s RETURNING id",
                (payload.status, appointment_id),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Cita no encontrada")
        conn.commit()
    return {"id": appointment_id, "status": payload.status}


@router.post("/appointments/optimize-live")
def optimize_appointment_with_live_db(payload: AppointmentOptimizePayload):
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, COALESCE(specialty, 'Medicina General') AS specialty, status FROM users WHERE role = 'Médico' AND status = 'Activo'"
            )
            live_doctors = cur.fetchall()

            cur.execute(
                """
                SELECT patient_id, doctor_id, doctor, TO_CHAR(date, 'YYYY-MM-DD') AS date, time, duration, status
                FROM appointments
                WHERE status != 'Cancelada' AND date >= CURRENT_DATE
                """
            )
            live_appointments = cur.fetchall()

    result = optimize_appointment_schedule(
        patient_id=payload.patient_id,
        patient_name=payload.patient_name,
        risk=payload.risk,
        specialty=payload.specialty,
        duration=payload.duration,
        preferred_date=payload.preferred_date,
        doctors=live_doctors if live_doctors else None,
        existing_appointments=live_appointments,
    )
    if result["recommended_slot"]:
        result["recommended_slot"]["reason"] = payload.reason
    return result


# ==============================================================================
# 5. AUDITORÍA DE SEGURIDAD Y HONEYPOT
# ==============================================================================
@router.get("/security/audit")
def list_audit_logs():
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, TO_CHAR(date, 'YYYY-MM-DD') AS date, time,
                       user_name AS "user", action, module, detail, ip, severity
                FROM audit_logs
                ORDER BY created_at DESC
                LIMIT 100
                """
            )
            return cur.fetchall()


@router.post("/security/audit")
def record_audit_log(payload: AuditCreateRequest, request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    return _write_audit_pg(payload.user, payload.action, payload.module, payload.detail, payload.severity, client_ip)


@router.get("/security/honeypot")
def list_honeypot_events():
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, timestamp, type, severity, route, detail,
                       user_name AS "user", role, origin,
                       user_agent AS "userAgent", attempted_user AS "attemptedUser"
                FROM honeypot_events
                ORDER BY timestamp DESC
                """
            )
            return cur.fetchall()


@router.post("/security/honeypot")
def record_honeypot_event(payload: HoneypotCreateRequest):
    hp_id = f"hp-{uuid.uuid4().hex[:8]}"
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO honeypot_events (
                    id, type, severity, route, detail, user_name, role, origin, user_agent, attempted_user
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    hp_id,
                    payload.type,
                    payload.severity,
                    payload.route,
                    payload.detail,
                    payload.user,
                    payload.role,
                    payload.origin,
                    payload.userAgent,
                    payload.attemptedUser,
                ),
            )
        conn.commit()

    _write_audit_pg(
        payload.user,
        "Intruso detectado en honeypot",
        "Seguridad",
        f"{payload.type}: {payload.detail}",
        "critical",
        payload.origin,
    )
    return {"id": hp_id, "status": "recorded"}