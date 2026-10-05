import os
import glob
import joblib
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Detección de Melanoma",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diagnóstico Prematuro de Melanoma")
st.write("Sube una imagen de una lesión cutánea para evaluar si es benigna o melanoma.")

MODEL_PATH = "modelo_reconstruido.pkl"

@st.cache_resource
def load_reconstructed_model():
    """Une los fragmentos del repositorio y los carga localmente con joblib."""
    if not os.path.exists(MODEL_PATH):
        # Buscar todas las partes guardadas en la raíz del proyecto
        parts = sorted(glob.glob("model_part_*.bin"))
        
        if not parts:
            raise FileNotFoundError("No se encontraron los fragmentos del modelo (model_part_*.bin) en el repositorio.")
            
        with open(MODEL_PATH, "wb") as outfile:
            for part in parts:
                with open(part, "rb") as infile:
                    outfile.write(infile.read())
                    
    return joblib.load(MODEL_PATH)

# Cargar el modelo
try:
    model = load_reconstructed_model()
except Exception as e:
    st.error(f"Error al cargar el archivo de modelo: {e}")
    st.stop()

# Interfaz de usuario
uploaded_file = st.file_uploader("Carga una imagen de lesión cutánea (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada para diagnóstico", use_container_width=True)
    
    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Evaluando la imagen con el modelo..."):
            img = image.convert('RGB').resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            try:
                prediction = model.predict(np.expand_dims(img_array, axis=0))
            except Exception:
                prediction = model.predict(img_array.flatten().reshape(1, -1))
            
            pred_val = float(prediction[0][0] if isinstance(prediction[0], (list, np.ndarray)) else prediction[0])
            
            if pred_val >= 0.5:
                st.error("⚠️ **Resultado:** Alta probabilidad de Melanoma detectada.")
            else:
                st.success("✅ **Resultado:** Posible Lesión Benigna.")
