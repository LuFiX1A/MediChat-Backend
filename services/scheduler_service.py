from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional

# Catálogo base sincronizado con los médicos de HealthText (respaldo mientras se conecta Supabase)
DEFAULT_DOCTORS: List[Dict[str, Any]] = [
    {
        "id": "u-001",
        "name": "Dra. Valentina Ortiz",
        "specialty": "Medicina General y Triaje Clínico",
        "keywords": ["general", "triaje", "quemadura", "urgencia", "preventivo", "control"],
        "status": "Activo",
    },
    {
        "id": "u-002",
        "name": "Dr. Camilo Rojas",
        "specialty": "Cardiología y Endocrinología",
        "keywords": ["cardiologia", "endocrinologia", "hipertension", "diabetes", "glucosa", "corazon", "toracico"],
        "status": "Activo",
    },
    {
        "id": "u-003",
        "name": "Dra. Natalia Cárdenas",
        "specialty": "Neumología",
        "keywords": ["neumologia", "asma", "respiratorio", "disnea", "pulmon", "tos"],
        "status": "Activo",
    },
    {
        "id": "u-006",
        "name": "Dr. Felipe Vargas",
        "specialty": "Gastroenterología",
        "keywords": ["gastroenterologia", "abdominal", "gastritis", "digestivo", "estomago"],
        "status": "Activo",
    },
]

RISK_WEIGHTS: Dict[str, float] = {
    "Alto": 10.0,
    "Medio": 4.0,
    "Bajo": 1.0,
}

RISK_PRIORITY_SCORE: Dict[str, int] = {
    "Alto": 3,
    "Medio": 2,
    "Bajo": 1,
}

WORK_START_MINUTES = 8 * 60    # 08:00
WORK_END_MINUTES = 18 * 60     # 18:00
SLOT_STEP_MINUTES = 30         # Bloques candidatos cada 30 min


def _time_to_minutes(time_str: str) -> int:
    """Convierte 'HH:MM' a minutos totales desde las 00:00."""
    hours, minutes = map(int, time_str.strip().split(":")[:2])
    return hours * 60 + minutes


def _minutes_to_time(total_minutes: int) -> str:
    """Convierte minutos totales desde las 00:00 a formato 'HH:MM'."""
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def _normalize_text(text: str) -> str:
    replacements = str.maketrans("áéíóúÁÉÍÓÚñÑ", "aeiouAEIOUnN")
    return text.lower().translate(replacements).strip()


def _specialty_match_score(doctor: Dict[str, Any], requested_specialty: str) -> float:
    """
    Calcula el costo g_specialty(n):
    - 0.0 si la especialidad coincide exactamente o por palabra clave del especialista.
    - 15.0 si es Medicina General (puede atender triaje/derivación).
    - 60.0 si pertenece a otra especialidad no relacionada.
    """
    req_norm = _normalize_text(requested_specialty)
    doc_spec_norm = _normalize_text(doctor.get("specialty", ""))
    keywords = [_normalize_text(k) for k in doctor.get("keywords", [])]

    if not req_norm or req_norm in doc_spec_norm or doc_spec_norm in req_norm:
        return 0.0

    if any(kw in req_norm or req_norm in kw for kw in keywords):
        return 0.0

    if "medicina general" in doc_spec_norm or "triaje" in doc_spec_norm:
        return 15.0

    return 60.0


def _has_time_collision(
    candidate_start: int,
    candidate_duration: int,
    booked_intervals: List[tuple],
) -> bool:
    """Verifica traslape estricto entre [start, end) y los intervalos ya ocupados."""
    candidate_end = candidate_start + candidate_duration
    for booked_start, booked_end in booked_intervals:
        if candidate_start < booked_end and candidate_end > booked_start:
            return True
    return False


def optimize_appointment_schedule(
    patient_id: str,
    patient_name: str,
    risk: str = "Medio",
    specialty: str = "Medicina General",
    duration: int = 30,
    preferred_date: Optional[str] = None,
    doctors: Optional[List[Dict[str, Any]]] = None,
    existing_appointments: Optional[List[Dict[str, Any]]] = None,
    horizon_days: int = 7,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Ejecuta búsqueda heurística informada f(n) = g(n) + h(n) sobre el espacio
    de estados (Médico x Fecha x Bloque Horario) y devuelve las top_k opciones óptimas.
    """
    active_doctors = [
        d for d in (doctors if doctors is not None else DEFAULT_DOCTORS)
        if d.get("status", "Activo") == "Activo"
    ]
    appointments = existing_appointments or []

    risk_clean = risk if risk in RISK_WEIGHTS else "Medio"
    severity_multiplier = RISK_WEIGHTS[risk_clean]
    priority_score = RISK_PRIORITY_SCORE[risk_clean]

    now = datetime.now()
    today = now.date()
    current_minutes_today = now.hour * 60 + now.minute + 30  # Margen mínimo de 30 min hoy

    if preferred_date:
        try:
            start_date = max(today, date.fromisoformat(preferred_date))
        except ValueError:
            start_date = today
    else:
        start_date = today

    # Indexar citas ocupadas por (doctor_id, fecha) y por (patient_id, fecha)
    doctor_busy: Dict[tuple, List[tuple]] = {}
    patient_busy: Dict[tuple, List[tuple]] = {}
    doctor_daily_load: Dict[tuple, int] = {}

    for appt in appointments:
        if appt.get("status") == "Cancelada":
            continue
        appt_date = str(appt.get("date", ""))
        appt_time = str(appt.get("time", "08:00"))
        appt_dur = int(appt.get("duration", 30))
        start_min = _time_to_minutes(appt_time)
        interval = (start_min, start_min + appt_dur)

        doc_id = appt.get("doctor_id") or appt.get("doctor")
        pat_id = appt.get("patient_id")

        if doc_id:
            doctor_busy.setdefault((doc_id, appt_date), []).append(interval)
            doctor_daily_load[(doc_id, appt_date)] = doctor_daily_load.get((doc_id, appt_date), 0) + 1
        if pat_id:
            patient_busy.setdefault((pat_id, appt_date), []).append(interval)

    candidates: List[Dict[str, Any]] = []

    for day_offset in range(horizon_days):
        target_date = start_date + timedelta(days=day_offset)
        target_date_str = target_date.isoformat()
        days_from_now = (target_date - today).days

        for doc in active_doctors:
            doc_id = doc.get("id", doc.get("name"))
            spec_cost = _specialty_match_score(doc, specialty)
            load_count = doctor_daily_load.get((doc_id, target_date_str), 0)
            load_cost = load_count * 4.0  # Penaliza saturación de agenda del médico

            doc_intervals = doctor_busy.get((doc_id, target_date_str), [])
            pat_intervals = patient_busy.get((patient_id, target_date_str), [])

            for slot_start in range(WORK_START_MINUTES, WORK_END_MINUTES - duration + 1, SLOT_STEP_MINUTES):
                if target_date == today and slot_start < current_minutes_today:
                    continue

                if _has_time_collision(slot_start, duration, doc_intervals):
                    continue
                if _has_time_collision(slot_start, duration, pat_intervals):
                    continue

                # g(n): Costo operativo (brecha de especialidad + carga del médico)
                g_cost = round(spec_cost + load_cost, 2)

                # h(n): Costo heurístico de espera clínica ponderado por riesgo
                wait_hours = (days_from_now * 24.0) + max(0.0, (slot_start - WORK_START_MINUTES) / 60.0)
                h_cost = round(wait_hours * severity_multiplier, 2)

                total_cost = round(g_cost + h_cost, 2)

                reasons = []
                if spec_cost == 0.0:
                    reasons.append(f"especialista compatible ({doc['specialty']})")
                elif spec_cost <= 15.0:
                    reasons.append("atención primaria de triaje disponible")
                if days_from_now == 0:
                    reasons.append("atención el mismo día")
                if load_count == 0:
                    reasons.append("agenda despejada")

                candidates.append({
                    "patient_id": patient_id,
                    "patient_name": patient_name,
                    "doctor_id": doc_id,
                    "doctor": doc["name"],
                    "specialty": doc["specialty"],
                    "date": target_date_str,
                    "time": _minutes_to_time(slot_start),
                    "duration": duration,
                    "risk": risk_clean,
                    "priority_score": priority_score,
                    "heuristic_metrics": {
                        "g_cost": g_cost,
                        "h_cost": h_cost,
                        "total_f_cost": total_cost,
                        "severity_weight": severity_multiplier,
                        "doctor_day_appointments": load_count,
                    },
                    "justification": (
                        f"Prioridad {risk_clean} (score {priority_score}/3): "
                        + (", ".join(reasons) if reasons else "primer bloque libre sin traslapes")
                        + "."
                    ),
                })

    candidates.sort(key=lambda item: item["heuristic_metrics"]["total_f_cost"])
    best_options = candidates[:top_k]

    return {
        "algorithm": "Informed Heuristic Search A* (f(n) = g(n) + h(n))",
        "evaluated_states": len(candidates),
        "recommended_slot": best_options[0] if best_options else None,
        "alternatives": best_options,
    }