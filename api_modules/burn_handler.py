from services.burn_service import process_burn_logic
from services.gemini_service import obtener_respuesta_gemini 

async def handle_integration_logic(file, user_text, doctors_list):
    # 1. Procesar la imagen con el h5
    img_analysis = await process_burn_logic(file)
    
    grado_detectado = img_analysis.get("grado", 1)
    confianza = img_analysis.get("confianza", 0)

    # 2. Llamada a Gemini mejorada
    # Le pasamos también la confianza para que Gemini sea más precavido si es baja
    respuesta_ia = await obtener_respuesta_gemini(
        mensaje_usuario=user_text,
        contexto_medico=f"Triaje inicial. La IA detectó Grado {grado_detectado} con {confianza}% de confianza.",
        texto_doctores_mongo=str(doctors_list),
        grado_ia=grado_detectado
    )

    return {
        "analisis_visual": img_analysis,
        "diagnostico_ia": respuesta_ia,
        "texto_usuario_procesado": user_text
    }