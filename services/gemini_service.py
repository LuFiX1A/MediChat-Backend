import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURACIÓN DE LA API ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Configuraciones de seguridad
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

generation_config = {
    "temperature": 0.4, 
    "top_p": 1, 
    "max_output_tokens": 2048,
    "response_mime_type": "application/json" 
}

# Se usa gemini-2.5-flash para máxima estabilidad en producción
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash", 
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Implementación de 'confianza_ia' como parámetro opcional para mayor contexto
async def obtener_respuesta_gemini(mensaje_usuario: str, contexto_medico: str, texto_doctores_mongo: str, grado_ia: int = None, confianza_ia: float = None):
    """
    Función central que genera la respuesta del chatbot integrando texto, 
    análisis visual y base de datos de doctores.
    """
    
    # 1. Construcción de la lógica de quemaduras
    info_quemadura = ""
    if grado_ia:
        pautas = {
            1: "Grado 1 (Epidérmica): Piel roja y seca. Lavar con agua fresca. No usar hielo.",
            2: "Grado 2 (Espesor parcial): Ampollas y dolor intenso. No reventar ampollas. Cubrir con gasa estéril.",
            3: "Grado 3 (Espesor total): Piel blanca o carbonizada. ¡URGENCIA MÉDICA! No retirar ropa pegada."
        }
        # Añadimos la confianza solo como contexto interno para Gemini
        conf_text = f" con un {confianza_ia}% de certeza" if confianza_ia else ""
        info_quemadura = f"\n[SISTEMA DE VISIÓN]: Se ha detectado una quemadura de {pautas.get(grado_ia, 'Grado desconocido')}{conf_text}."
    else:
        info_quemadura = "\n[SISTEMA DE VISIÓN]: No se proporcionó imagen o no se detectó una quemadura específica."

    # 2. PROMPT FINAL (ESTRUCTURA BASE INTACTA)
    prompt_final = f"""
    Eres MediChat, un asistente virtual de triaje médico experto. Tu objetivo es orientar al usuario de forma clara, segura y profesional.

    CONTEXTO DEL CASO:
    - Paciente describe: "{mensaje_usuario}"
    - Entorno: {contexto_medico}
    {info_quemadura}

    BASE DE DATOS DE DOCTORES DISPONIBLES (MongoDB):
    {texto_doctores_mongo}

    INSTRUCCIONES DE RESPUESTA:
    1. Evalúa si la consulta es médica. Si no lo es, declina amablemente.
    2. Si hay una quemadura (Grado 1, 2 o 3), da las pautas de primeros auxilios específicas mencionadas arriba.
    3. Recomienda OBLIGATORIAMENTE al doctor más adecuado de la lista según la especialidad (ej: si es quemadura, busca Dermatólogos o especialistas en Urgencias).
    4. Proporciona el ID del doctor para que el sistema genere la tarjeta de contacto.
    5. Mantén un tono empático pero directo.

    FORMATO DE SALIDA (JSON ESTRICTO):
    {{
        "es_medico": true,
        "mensaje_al_usuario": "Texto principal de tu respuesta aquí...",
        "pautas_inmediatas": ["Paso 1", "Paso 2"],
        "recomendaciones": [
            {{
                "nombre_doctor": "Nombre completo",
                "especialidad": "Especialidad",
                "id_mongo": "ID del doctor"
            }}
        ]
    }}
    """

    try:
        response = await model.generate_content_async(prompt_final)
        return json.loads(response.text.strip())
    except Exception as e:
        print(f"Error en Gemini Service: {e}")
        return {
            "es_medico": False, 
            "mensaje_al_usuario": "Lo siento, tuve un problema técnico al procesar tu consulta médica.",
            "pautas_inmediatas": [],
            "recomendaciones": []
        }