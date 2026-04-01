# MediChat - Backend Intelligence Engine 🧠🔥

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Uvicorn](https://img.shields.io/badge/Server-Uvicorn-444444?logo=python)](https://www.uvicorn.org/)

## 📝 Descripción Técnica
Este repositorio contiene el núcleo de procesamiento de **MediChat**. Se encarga de la recepción de imágenes y texto, la inferencia mediante Redes Neuronales Convolucionales (CNN) y la generación de diagnósticos preliminares automatizados.

El sistema utiliza un modelo de aprendizaje profundo para clasificar lesiones cutáneas en tres niveles de gravedad, integrando una capa de lógica de seguridad para minimizar errores de diagnóstico.

---

## 🛠️ Arquitectura y Tecnologías
- **Motor de IA:** Red Neuronal Convolucional (CNN) desarrollada en **Keras/TensorFlow**.
- **API Framework:** **FastAPI** para una comunicación asíncrona de alto rendimiento.
- **Procesamiento de Datos:** NumPy y Pillow para la transformación de imágenes a tensores.
- **Túnel de Red:** Configuración optimizada para **Ngrok**, permitiendo el despliegue de inferencia local hacia dispositivos móviles.

---

## 🧠 Lógica de Análisis de IA
Para garantizar la integridad de los resultados, el backend implementa las siguientes reglas:

1. **Preprocesamiento:** Las imágenes se normalizan y redimensionan a los requisitos de entrada del modelo.
2. **Umbral de Confianza (Threshold):** Se ha establecido un **umbral mínimo del 70% (0.70)**. 
   - Si la confianza de la predicción es inferior a este valor, el sistema retorna un **Grado 0 (No concluyente)**.
   - Esta medida de seguridad evita falsos positivos ante imágenes de ruido o contenido no médico.
3. **Respuesta Unificada:** Se genera un JSON estructurado que incluye el grado, el porcentaje de confianza, el detalle médico y la lista de recomendaciones.

---

## 🚀 Instalación y Despliegue

### Configuración del Backend
1. Clonar repositorio y crear entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate

2. Instalar Dependencias:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   
3. Exponer vía Ngrok:
   ```bash
   ngrok http 8000

---

## 📌 Endpoints Principales
POST /analyze-full: Recibe mensaje (Texto) y file (Imagen). Retorna el análisis completo de triaje.

   
