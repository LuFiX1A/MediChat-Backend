# api_modules/burn_handler.py
from services.burn_service import process_burn_logic
from services.gemini_service import obtener_respuesta_gemini 

async def handle_integration_logic(file, user_text, doctors_list):
    # 1. Procesar la imagen (Llama a tu simulación o modelo .h5)
    img_analysis = await process_burn_logic(file)
    
    # CORRECCIÓN DE LLAVE: 
    # Accedemos directo a 'grado' porque tu burn_service devuelve {"grado": ...}
    grado_detectado = img_analysis.get("grado", 1) 

    # 2. Llamada a Gemini
    # Enviamos los datos crudos. Gemini se encarga de armar la respuesta 
    # profesional usando su propio template interno.
    respuesta_ia = await obtener_respuesta_gemini(
        mensaje_usuario=user_text,
        contexto_medico="Triaje inicial de urgencias",
        texto_doctores_mongo=str(doctors_list),
        grado_ia=grado_detectado
    )

    # 3. Respuesta unificada para el Frontend
    return {
        "analisis_visual": img_analysis,
        "diagnostico_ia": respuesta_ia,
        "texto_usuario_procesado": user_text
    }