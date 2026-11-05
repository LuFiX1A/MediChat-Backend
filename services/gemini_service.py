import os
import json
from typing import Optional, Dict, Any, List
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

# Implementación de 'ai_confidence' como parámetro opcional para mayor contexto
async def get_gemini_response(
    user_message: str = "",
    medical_context: str = "Consulta Médica General",
    mongo_doctors_text: str = "[]",
    ai_grade: Optional[int] = None,
    ai_confidence: Optional[float] = None,
    **kwargs
):
    """
    Función central que genera la respuesta del chatbot integrando texto, 
    análisis visual y base de datos de doctores.
    Central function generating chatbot response integrating text, visual analysis and doctor database.
    """
    # Soporte para argumentos heredados en español (Backwards compatibility)
    user_message = user_message or kwargs.get("mensaje_usuario", "")
    medical_context = kwargs.get("contexto_medico", medical_context)
    mongo_doctors_text = kwargs.get("texto_doctores_mongo", mongo_doctors_text)
    if ai_grade is None:
        ai_grade = kwargs.get("grado_ia")
    if ai_confidence is None:
        ai_confidence = kwargs.get("confianza_ia")

    # 1. Construcción de la lógica de quemaduras (Burn logic construction)
    burn_info = ""
    if ai_grade:
        guidelines = {
            1: "Grado 1 (Epidérmica): Piel roja y seca. Lavar con agua fresca. No usar hielo.",
            2: "Grado 2 (Espesor parcial): Ampollas y dolor intenso. No reventar ampollas. Cubrir con gasa estéril.",
            3: "Grado 3 (Espesor total): Piel blanca o carbonizada. ¡URGENCIA MÉDICA! No retirar ropa pegada."
        }
        # Añadimos la confianza solo como contexto interno para Gemini
        confidence_text = f" con un {ai_confidence}% de certeza" if ai_confidence else ""
        burn_info = f"\n[SISTEMA DE VISIÓN]: Se ha detectado una quemadura de {guidelines.get(ai_grade, 'Grado desconocido')}{confidence_text}."
    else:
        burn_info = "\n[SISTEMA DE VISIÓN]: No se proporcionó imagen o no se detectó una quemadura específica."

    # 2. PROMPT FINAL (ESTRUCTURA BASE INTACTA)
    final_prompt = f"""
    Eres MediChat, un asistente virtual de triaje médico experto. Tu objetivo es orientar al usuario de forma clara, segura y profesional.

    CONTEXTO DEL CASO:
    - Paciente describe: "{user_message}"
    - Entorno: {medical_context}
    {burn_info}

    BASE DE DATOS DE DOCTORES DISPONIBLES (MongoDB):
    {mongo_doctors_text}

    INSTRUCCIONES DE RESPUESTA:
    1. Evalúa si la consulta es médica. Si no lo es, declina amablemente.
    2. Si hay una quemadura (Grado 1, 2 o 3), da las pautas de primeros auxilios específicas mencionadas arriba.
    3. Recomienda OBLIGATORIAMENTE al doctor más adecuado de la lista según la especialidad (ej: si es quemadura, busca Dermatólogos o especialistas en Urgencias).
    4. Proporciona el ID del doctor para que el sistema genere la tarjeta de contacto.
    5. Mantén un tono empático pero directo.

    FORMATO DE SALIDA (JSON ESTRICTO):
    {{
        "is_medical": true,
        "user_message": "Texto principal de tu respuesta aquí...",
        "immediate_guidelines": ["Paso 1", "Paso 2"],
        "recommendations": [
            {{
                "doctor_name": "Nombre completo",
                "specialty": "Especialidad",
                "mongo_id": "ID del doctor"
            }}
        ]
    }}
    """

    try:
        response = await model.generate_content_async(final_prompt)
        raw_response = json.loads(response.text.strip())

        # Sincronización bidireccional entre claves en inglés y español para total compatibilidad
        is_medical = raw_response.get("is_medical", raw_response.get("es_medico", False))
        user_msg = raw_response.get("user_message", raw_response.get("mensaje_al_usuario", ""))
        immediate_guidelines = raw_response.get("immediate_guidelines", raw_response.get("pautas_inmediatas", []))
        raw_recommendations = raw_response.get("recommendations", raw_response.get("recomendaciones", []))

        normalized_recommendations = []
        for rec in raw_recommendations:
            doc_name = rec.get("doctor_name", rec.get("nombre_doctor", ""))
            specialty = rec.get("specialty", rec.get("especialidad", ""))
            doc_id = rec.get("mongo_id", rec.get("id_mongo", ""))
            normalized_recommendations.append({
                "doctor_name": doc_name,
                "specialty": specialty,
                "mongo_id": doc_id,
                # Claves en español para compatibilidad
                "nombre_doctor": doc_name,
                "especialidad": specialty,
                "id_mongo": doc_id
            })

        return {
            "is_medical": is_medical,
            "user_message": user_msg,
            "immediate_guidelines": immediate_guidelines,
            "recommendations": normalized_recommendations,
            # Claves en español para compatibilidad con el frontend de Ionic
            "es_medico": is_medical,
            "mensaje_al_usuario": user_msg,
            "pautas_inmediatas": immediate_guidelines,
            "recomendaciones": normalized_recommendations
        }
    except Exception as e:
        print(f"Error en Gemini Service: {e}")
        return {
            "is_medical": False,
            "user_message": "Lo siento, tuve un problema técnico al procesar tu consulta médica.",
            "immediate_guidelines": [],
            "recommendations": [],
            # Claves en español para compatibilidad
            "es_medico": False, 
            "mensaje_al_usuario": "Lo siento, tuve un problema técnico al procesar tu consulta médica.",
            "pautas_inmediatas": [],
            "recomendaciones": []
        }

# Alias para compatibilidad hacia atrás
obtener_respuesta_gemini = get_gemini_response