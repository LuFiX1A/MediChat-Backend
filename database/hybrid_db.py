import os
import psycopg
from psycopg.rows import dict_row
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "")
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB_V2 = os.getenv("MONGO_DB_V2", "HealthTextV2")

# --- CONEXIÓN MONGODB ATLAS (HealthTextV2) ---
mongo_client = MongoClient(MONGO_URI) if MONGO_URI else None
db_v2 = mongo_client[MONGO_DB_V2] if mongo_client is not None else None

doctors_col = db_v2["doctors"] if db_v2 is not None else None
records_col = db_v2["medical_records"] if db_v2 is not None else None
triage_col = db_v2["triage_sessions"] if db_v2 is not None else None


# --- CONEXIÓN POSTGRESQL (SUPABASE) ---
def get_pg_connection():
    """Abre una conexión directa a Supabase PostgreSQL retornando filas como diccionarios."""
    if not SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL no está configurada en el archivo .env")
    return psycopg.connect(SUPABASE_DB_URL, row_factory=dict_row)


def check_hybrid_health() -> dict:
    """Verifica el estado de conectividad de ambos motores en la nube."""
    status = {"postgresql_supabase": "disconnected", "mongodb_healthtext_v2": "disconnected"}
    try:
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                status["postgresql_supabase"] = "connected"
    except Exception as e:
        status["postgresql_supabase"] = f"error: {str(e)}"

    try:
        if mongo_client is not None:
            mongo_client.admin.command("ping")
            status["mongodb_healthtext_v2"] = "connected"
    except Exception as e:
        status["mongodb_healthtext_v2"] = f"error: {str(e)}"

    return status