import os
import urllib.request
import joblib
import numpy as np
import streamlit as st
from PIL import Image

# Configuración de página en Streamlit
st.set_page_config(
    page_title="Detección de Melanoma",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diagnóstico Prematuro de Melanoma")
st.write("Sube una imagen de una lesión cutánea para evaluar si es benigna o melanoma.")

# URL de GitHub Release
MODEL_URL = "https://github.com/SanRonal10/melanoma2/releases/download/v1.0.0/mimodelo.pkl"
MODEL_FILENAME = "mimodelo.pkl"

@st.cache_resource
def load_model():
    """Descarga el archivo del modelo desde GitHub Release y lo carga en memoria."""
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Descargando el modelo de diagnóstico desde GitHub Releases..."):
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(MODEL_URL, headers=headers)
            with urllib.request.urlopen(req) as response, open(MODEL_FILENAME, 'wb') as out_file:
                out_file.write(response.read())

    return joblib.load(MODEL_FILENAME)

# Carga segura del modelo
try:
    model = load_model()
except Exception as e:
    if os.path.exists(MODEL_FILENAME):
        os.remove(MODEL_FILENAME)
    st.error(
        f"Error al cargar el archivo de modelo: {e}. "
        "El archivo corrupto se ha eliminado. Por favor vuelve a cargar la página."
    )
    st.stop()

def preprocess_image(image):
    """Preprocesa la imagen de entrada ajustando tamaño y normalizando píxeles."""
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    # Formato tensor 4D y formato plano 2D para compatibilidad con distintos modelos
    tensor_features = np.expand_dims(img_array / 255.0, axis=0)
    flat_features = img_array.flatten().reshape(1, -1)
    
    return tensor_features, flat_features

# Módulo de carga de imágenes
uploaded_file = st.file_uploader("Carga una imagen de lesión cutánea (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada para diagnóstico", use_container_width=True)
    
    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Procesando la imagen con el modelo de clasificación..."):
            tensor_img, flat_img = preprocess_image(image)
            
            try:
                prediction = model.predict(tensor_img)
            except Exception:
                prediction = model.predict(flat_img)
            
            # Evaluación de la predicción retornada
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
