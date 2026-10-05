import os
import requests
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Detección de Melanoma",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diagnóstico Prematuro de Melanoma")
st.write("Sube una imagen de una lesión cutánea para evaluar si es benigna o melanoma.")

MODEL_URL = "https://github.com/SanRonal10/melanoma2/releases/download/v1.0.0/mimodelo.pkl"
LOCAL_MODEL_PATH = "mimodelo_local.h5"

@st.cache_resource
def load_keras_model():
    if not os.path.exists(LOCAL_MODEL_PATH) or os.path.getsize(LOCAL_MODEL_PATH) < 10000:
        if os.path.exists(LOCAL_MODEL_PATH):
            os.remove(LOCAL_MODEL_PATH)
            
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(MODEL_URL, headers=headers, allow_redirects=True, stream=True)
        
        if response.status_code == 200:
            with open(LOCAL_MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            raise RuntimeError(f"Error HTTP {response.status_code} al descargar el modelo.")

    # Cargar usando Keras en lugar de joblib
    return tf.keras.models.load_model(LOCAL_MODEL_PATH)

try:
    model = load_keras_model()
except Exception as e:
    st.error(f"Error al cargar el archivo de modelo: {e}")
    st.stop()

def preprocess_image(image):
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

uploaded_file = st.file_uploader("Carga una imagen de lesión cutánea (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada para diagnóstico", use_container_width=True)
    
    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Evaluando la imagen..."):
            input_data = preprocess_image(image)
            prediction = model.predict(input_data)
            
            pred_num = float(prediction[0][0]) if isinstance(prediction[0], (list, np.ndarray)) else float(prediction[0])
            
            if pred_num >= 0.5:
                st.error(f"⚠️ **Resultado:** Alta probabilidad de Melanoma detectada. ({pred_num*100:.1f}%)")
            else:
                st.success(f"✅ **Resultado:** Posible Lesión Benigna. ({(1-pred_num)*100:.1f}%)")
