import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURACIÓN TÉCNICA (Mudada del Main) ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Mantenemos tus configuraciones de seguridad
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

generation_config = {
    "temperature": 0.5,
    "top_p": 1, 
    "max_output_tokens": 2048,
    "response_mime_type": "application/json" 
}

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash", 
    generation_config=generation_config,
    safety_settings=safety_settings
)

async def obtener_respuesta_gemini(mensaje_usuario: str, contexto_medico: str, texto_doctores_mongo: str, grado_ia: int = None):
    # Lógica de quemaduras (Se inyecta solo si existe grado_ia)
    info_quemadura = ""
    if grado_ia is not None:
        pautas = {
            1: "Grado 1: Lavar con agua fresca. No usar remedios caseros.",
            2: "Grado 2: No reventar ampollas, cubrir con gasa. Consultar especialista.",
            3: "Grado 3: ¡URGENCIA! No quitar ropa pegada y acudir a emergencias."
        }
        info_quemadura = f"\n[ANALISIS IA - QUEMADURA GRADO {grado_ia}]: {pautas.get(grado_ia)}"
    else:
        # Escenario cuando NO se detecta quemadura o no hay imagen
        info_quemadura = "\n[SISTEMA]: No se dispone de análisis de imagen de quemadura para esta consulta."

    prompt_final = f"""
    Eres MediChat, un asistente médico de triaje inteligente especializado en quemaduras y red médica.
    
    TUS DATOS (DOCTORES DISPONIBLES):
    {texto_doctores_mongo}
    
    {info_quemadura}

    INPUT DEL USUARIO:
    - Historial/Contexto: {contexto_medico}
    - Mensaje actual: "{mensaje_usuario}"
    
    INSTRUCCIONES CRÍTICAS:
    1. Evalúa si el caso es médico o relacionado con salud.
    2. Si el usuario habla de temas NO MÉDICOS (ocio, política, bromas, etc.), responde que como asistente médico no puedes atender esa solicitud.
    3. Si el usuario pregunta por una quemadura pero NO hay 'grado_ia' detectado, indícale que para un mejor triaje puede subir una foto, pero da consejos generales de prevención.
    4. Si hay grado de quemadura, intégralo con prioridad absoluta.
    5. Recomienda al doctor más apto de la lista.
    
    FORMATO DE RESPUESTA JSON (OBLIGATORIO):
    {{
        "es_medico": true/false,
        "mensaje_al_usuario": "Tu respuesta aquí...",
        "recomendaciones": [] 
    }}
    """

    try:
        response = await model.generate_content_async(prompt_final)
        return json.loads(response.text.strip())
    except Exception as e:
        return {
            "es_medico": False, 
            "mensaje_al_usuario": "Lo siento, no puedo procesar esta solicitud en este momento.",
            "recomendaciones": []
        }