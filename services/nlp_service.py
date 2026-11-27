import os
import re
import json
import joblib
import numpy as np
from typing import Dict, Any, List

MODEL_PATH = "modelo_nlp_eme.joblib"
METRICS_PATH = "nlp_metrics.json"

nlp_model = None
nlp_metrics_cache: Dict[str, Any] = {}

# Cargar modelo al iniciar el servicio
try:
    if os.path.exists(MODEL_PATH):
        nlp_model = joblib.load(MODEL_PATH)
        print("✅ Modelo NLP (Minería de Texto) cargado y listo")
    else:
        print(f"⚠️ Advertencia: No se encontró {MODEL_PATH} en la raíz del proyecto")
except Exception as e:
    print(f"❌ Error crítico cargando el modelo NLP: {e}")

# Cargar métricas de validación (Precision, Recall, F1-Score)
try:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            nlp_metrics_cache = json.load(f)
        print("✅ Métricas NLP (Precision/Recall/F1) cargadas correctamente")
except Exception as e:
    print(f"⚠️ No se pudo leer {METRICS_PATH}: {e}")


def clean_clinical_text(text: str) -> str:
    """Normaliza el texto clínico manteniendo números y negaciones médicas."""
    text = str(text).lower()
    text = re.sub(r"[^a-záéíóúñü0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_top_keywords(cleaned_text: str, top_n: int = 5) -> List[str]:
    """Extrae los términos con mayor peso TF-IDF dentro de la nota clínica."""
    if nlp_model is None or not cleaned_text:
        return []
    try:
        vectorizer = nlp_model.named_steps.get("tfidf")
        if vectorizer is None:
            return []
        tfidf_matrix = vectorizer.transform([cleaned_text])
        feature_names = np.array(vectorizer.get_feature_names_out())
        scores = tfidf_matrix.toarray()[0]
        non_zero_indices = np.where(scores > 0)[0]
        if len(non_zero_indices) == 0:
            return []
        sorted_indices = non_zero_indices[np.argsort(scores[non_zero_indices])[::-1]]
        return feature_names[sorted_indices[:top_n]].tolist()
    except Exception:
        return []


def analyze_clinical_note(
    symptoms: str = "",
    notes: str = "",
    diagnosis: str = ""
) -> Dict[str, Any]:
    """
    Analiza texto clínico no estructurado y devuelve:
    - predicted_risk: 'Bajo' | 'Medio' | 'Alto'
    - confidence: float (0.0 a 1.0)
    - probabilities: desglose por clase
    - extracted_keywords: términos clínicos clave
    """
    raw_combined = f"{symptoms} {notes} {diagnosis}".strip()
    cleaned = clean_clinical_text(raw_combined)

    if nlp_model is None or not cleaned:
        return {
            "predicted_risk": "Medio",
            "confidence": 0.0,
            "probabilities": {"Bajo": 0.0, "Medio": 0.0, "Alto": 0.0},
            "extracted_keywords": [],
            "status": "fallback"
        }

    predicted_class = str(nlp_model.predict([cleaned])[0])
    probs = nlp_model.predict_proba([cleaned])[0]
    classes = list(nlp_model.classes_)

    prob_map = {
        str(cls): round(float(prob), 4)
        for cls, prob in zip(classes, probs)
    }
    confidence = round(float(np.max(probs)), 4)
    keywords = extract_top_keywords(cleaned, top_n=5)

    return {
        "predicted_risk": predicted_class,
        "confidence": confidence,
        "probabilities": prob_map,
        "extracted_keywords": keywords,
        "status": "ok"
    }


def get_nlp_evaluation_metrics() -> Dict[str, Any]:
    """Retorna las métricas oficiales de Precision, Recall, F1-Score y Matriz de Confusión."""
    if not nlp_metrics_cache:
        return {"error": "Métricas NLP no disponibles en el servidor"}
    return nlp_metrics_cache