import os
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from urllib.request import urlretrieve

# Configuración de página
st.set_page_config(
    page_title="Detección de Melanoma",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diagnóstico Prematuro de Melanoma")
st.write("Sube una imagen de una lesión cutánea para evaluar si es benigna o melanoma.")

MODEL_URL = "https://github.com/SanRonal10/melanoma2/releases/download/v1.0.0/mimodelo.pkl"
LOCAL_MODEL_PATH = "mimodelo.pkl"

@st.cache_resource
def load_model():
    """Descarga el archivo pickle de GitHub Release de forma directa y lo carga con joblib."""
    if not os.path.exists(LOCAL_MODEL_PATH) or os.path.getsize(LOCAL_MODEL_PATH) < 10000:
        if os.path.exists(LOCAL_MODEL_PATH):
            os.remove(LOCAL_MODEL_PATH)
        # urlretrieve maneja automáticamente las redirecciones HTTP de GitHub Releases
        urlretrieve(MODEL_URL, LOCAL_MODEL_PATH)

    return joblib.load(LOCAL_MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    # Limpiar archivo en caso de descarga incompleta
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            os.remove(LOCAL_MODEL_PATH)
        except Exception:
            pass
    st.error(f"Error al cargar el archivo de modelo: {e}. Por favor vuelve a cargar la página.")
    st.stop()

def preprocess_image(image):
    """Preprocesa la imagen ajustando dimensiones y aplanando características."""
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    # Devuelve tanto la versión 4D como la aplanada 2D según lo requiera tu modelo
    tensor_features = np.expand_dims(img_array / 255.0, axis=0)
    flat_features = img_array.flatten().reshape(1, -1)
    
    return tensor_features, flat_features

uploaded_file = st.file_uploader("Carga una imagen de lesión cutánea (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada para diagnóstico", use_container_width=True)
    
    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Evaluando la imagen con el modelo..."):
            tensor_img, flat_img = preprocess_image(image)
            
            try:
                prediction = model.predict(flat_img)
            except Exception:
                prediction = model.predict(tensor_img)
            
            # Formatear la salida de predicción
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
