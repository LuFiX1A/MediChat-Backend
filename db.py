"""Capa de acceso a las bases de datos de HealthText.

PostgreSQL (Supabase) se usa mediante psycopg y MongoDB Atlas mediante pymongo.
Toda la configuración se lee de las variables de entorno del archivo .env.
"""
import os

import psycopg
from psycopg.rows import dict_row
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DATABASE = "HealthTextV2"

mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000) if MONGO_URI else None
mongo_db = mongo_client[MONGO_DATABASE] if mongo_client is not None else None


def pg_connection():
    """Abre una conexión nueva a Supabase. El caller debe usarla con `with`."""
    if not SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL no está configurada en .env")
    return psycopg.connect(SUPABASE_DB_URL, row_factory=dict_row)


def pg_all(sql: str, params: tuple = ()):
    """Devuelve todas las filas como lista de diccionarios."""
    with pg_connection() as conn:
        return conn.execute(sql, params).fetchall()


def pg_one(sql: str, params: tuple = ()):
    """Devuelve una fila como diccionario o None."""
    with pg_connection() as conn:
        return conn.execute(sql, params).fetchone()
