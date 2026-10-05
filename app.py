import os
import requests
import joblib
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Detección de Melanoma", page_icon="🩺", layout="centered")

st.title("🩺 Diagnóstico Prematuro de Melanoma")
st.write("Sube una imagen de una lesión cutánea para evaluar si es benigna o melanoma.")

# URL de descarga directa
MODEL_URL = "https://github.com/SanRonal10/melanoma2/releases/download/v1.0.0/mimodelo.pkl"
LOCAL_PATH = "mimodelo_validado.pkl"

@st.cache_resource
def load_model_safely():
    # Descargar si el archivo no existe o pesa menos de 1 MB
    if not os.path.exists(LOCAL_PATH) or os.path.getsize(LOCAL_PATH) < 1000000:
        if os.path.exists(LOCAL_PATH):
            os.remove(LOCAL_PATH)

        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        response = session.get(MODEL_URL, headers=headers, allow_redirects=True, stream=True)
        
        if response.status_code != 200:
            raise ValueError(f"Error HTTP {response.status_code} al acceder a la URL del modelo.")
            
        # Verificar que la respuesta sea binaria y no HTML
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise ValueError("La URL devolvió una página HTML en lugar del archivo binario .pkl.")

        with open(LOCAL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

    # Validar que el archivo descargado no empiece con etiquetas HTML
    with open(LOCAL_PATH, "rb") as f:
        header = f.read(10)
        if header.startswith(b"<!DOCTYPE") or header.startswith(b"<html"):
            os.remove(LOCAL_PATH)
            raise ValueError("El archivo descargado es una página HTML corrupta. Verifica que el Release sea público.")

    return joblib.load(LOCAL_PATH)

try:
    model = load_model_safely()
except Exception as e:
    st.error(f"Error al cargar el archivo de modelo: {e}")
    st.stop()

# Preprocesamiento e Interfaz
uploaded_file = st.file_uploader("Carga una imagen de lesión cutánea (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)
    
    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Procesando..."):
            img = image.convert('RGB').resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            try:
                pred = model.predict(np.expand_dims(img_array, axis=0))
            except Exception:
                pred = model.predict(img_array.flatten().reshape(1, -1))
                
            pred_val = float(pred[0][0] if isinstance(pred[0], (list, np.ndarray)) else pred[0])
            
            if pred_val >= 0.5:
                st.error("⚠️ **Resultado:** Alta probabilidad de Melanoma detectada.")
            else:
                st.success("✅ **Resultado:** Posible Lesión Benigna.")
