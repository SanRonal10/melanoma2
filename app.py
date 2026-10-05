import os
import requests
import joblib
import numpy as np
import streamlit as st
from PIL import Image

# Configuración de página
st.set_page_config(
    page_title="Detección de Melanoma",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diagnóstico Prematuro de Melanoma")
st.write("Sube una imagen de una lesión cutánea para evaluar si es benigna o melanoma.")

MODEL_URL = "https://github.com/SanRonal10/melanoma2/releases/download/v1.0.0/mimodelo.pkl"
MODEL_FILENAME = "mimodelo.pkl"

@st.cache_resource
def load_model_from_github():
    """Descarga el archivo descargando correctamente las redirecciones de GitHub Release."""
    # Descargar únicamente si no existe o si el archivo guardado está incompleto
    if not os.path.exists(MODEL_FILENAME) or os.path.getsize(MODEL_FILENAME) < 10000:
        if os.path.exists(MODEL_FILENAME):
            os.remove(MODEL_FILENAME)

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(MODEL_URL, headers=headers, allow_redirects=True, stream=True)
        
        if response.status_code == 200:
            with open(MODEL_FILENAME, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            raise RuntimeError(f"Error al descargar el modelo. Código HTTP: {response.status_code}")

    # Cargar usando la ruta de archivo str (nunca BytesIO)
    return joblib.load(MODEL_FILENAME)

# Carga del modelo
try:
    model = load_model_from_github()
except Exception as e:
    if os.path.exists(MODEL_FILENAME):
        try:
            os.remove(MODEL_FILENAME)
        except Exception:
            pass
    st.cache_resource.clear()
    st.error(f"Error al cargar el archivo de modelo: {e}. Por favor vuelve a cargar la página.")
    st.stop()

def preprocess_image(image):
    """Ajusta dimensiones y normaliza píxeles de la imagen."""
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    tensor_features = np.expand_dims(img_array / 255.0, axis=0)
    flat_features = img_array.flatten().reshape(1, -1)
    
    return tensor_features, flat_features

# Interfaz de carga
uploaded_file = st.file_uploader("Carga una imagen de lesión cutánea (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada para diagnóstico", use_container_width=True)
    
    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Evaluando la imagen con el modelo..."):
            tensor_img, flat_img = preprocess_image(image)
            
            try:
                prediction = model.predict(tensor_img)
            except Exception:
                prediction = model.predict(flat_img)
            
            if isinstance(prediction, (list, np.ndarray)):
                pred_val = prediction[0]
                if isinstance(pred_val, (list, np.ndarray)):
                    pred_val = pred_val[0]
            else:
                pred_val = prediction
                
            pred_num = float(pred_val)
            
            if pred_num >= 0.5 or pred_num == 1:
                st.error("⚠️ **Resultado:** Alta probabilidad de Melanoma detectada.")
            else:
                st.success("✅ **Resultado:** Posible Lesión Benigna.")
