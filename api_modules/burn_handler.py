# api_modules/burn_handler.py
from services.burn_service import process_burn_logic  # IMPORTAS lo del jefe
from services.gemini_services import call_gemini      # IMPORTAS el motor de IA
import json

async def handle_integration_logic(file, user_text, doctors_list):
    """
    ¿Qué hace?: Une el análisis de imagen del jefe con el texto de tu Front.
    ¿Para qué sirve?: Para dar una respuesta personalizada sin tocar el main.py.
    """
    
    # 1. LLAMAS A LA LÓGICA DEL JEFE (Para guardar la foto y obtener el grado)
    # No modificas su archivo, solo lo usas.
    img_analysis = await process_burn_logic(file)
    grado_ia = img_analysis.get("grado", 1) # Extraes el grado que su código calculó

    # 2. CREAS EL PROMPT PERSONALIZADO (Tu aporte intelectual)
    prompt_medico = f"""
    Eres un experto en triaje. 
    Caso: El usuario describe "{user_text}".
    Resultado IA: La imagen sugiere grado {grado_ia}.
    Doctores: {doctors_list}
    
    Instrucción: Si el texto dice 'no siento nada' y la IA dice 'grado 3', prioriza URGENCIA ALTA.
    Responde en JSON con: mensaje, pautas_higiene y id_doctor.
    """

    # 3. LLAMAS AL SERVICIO DE GEMINI DEL JEFE
    respuesta_ia_raw = await call_gemini(prompt_medico)
    
    # 4. UNES TODO Y ENTREGAS AL FRONTEND (IONIC)
    return {
        "analisis_visual": img_analysis,
        "diagnostico_ia": respuesta_ia_raw,
        "texto_usuario_procesado": user_text
    }