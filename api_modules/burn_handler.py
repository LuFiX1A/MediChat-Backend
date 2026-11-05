from services.burn_service import process_burn_logic
from services.gemini_service import get_gemini_response, obtener_respuesta_gemini 

async def handle_integration_logic(file, user_text: str, doctors_list: list):
    # 1. Procesar la imagen con el modelo CNN (Process image with CNN model)
    img_analysis = await process_burn_logic(file)
    
    detected_grade = img_analysis.get("grade", img_analysis.get("grado", 1))
    confidence = img_analysis.get("confidence", img_analysis.get("confianza", 0))

    # 2. Llamada a Gemini mejorada con grado y nivel de confianza (Enhanced Gemini call)
    ai_response = await get_gemini_response(
        user_message=user_text,
        medical_context=f"Triaje inicial. La IA detectó Grado {detected_grade} con {confidence}% de confianza.",
        mongo_doctors_text=str(doctors_list),
        ai_grade=detected_grade,
        ai_confidence=confidence
    )

    return {
        "visual_analysis": img_analysis,
        "ai_diagnosis": ai_response,
        "processed_user_text": user_text,
        "burn_grade": detected_grade,
        # Claves para compatibilidad total con el frontend existente (Ionic / Web)
        "analisis_visual": img_analysis,
        "diagnostico_ia": ai_response,
        "texto_usuario_procesado": user_text,
        "grado_quemadura": detected_grade
    }