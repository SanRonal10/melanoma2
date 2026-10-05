import os
import urllib.request
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

def download_file_if_missing():
    """Descarga el archivo físico si no existe o si el archivo local está vacío/corrupto."""
    need_download = False

    if not os.path.exists(MODEL_FILENAME):
        need_download = True
    elif os.path.getsize(MODEL_FILENAME) < 1000:  # Si pesa menos de 1 KB, está corrupto o es HTML
        os.remove(MODEL_FILENAME)
        need_download = True

    if need_download:
        with st.spinner("Descargando el modelo de diagnóstico desde GitHub Releases..."):
            req = urllib.request.Request(
                MODEL_URL,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response:
                content = response.read()
                with open(MODEL_FILENAME, 'wb') as f:
                    f.write(content)

@st.cache_resource
def load_model():
    """Garantiza la descarga física antes de invocar joblib.load con la ruta del archivo."""
    download_file_if_missing()
    # Importante: pasar el string con la ruta del archivo, no un objeto BytesIO
    return joblib.load(MODEL_FILENAME)

# Carga del modelo con gestión de excepciones y limpieza automática
try:
    model = load_model()
except Exception as e:
    if os.path.exists(MODEL_FILENAME):
        try:
            os.remove(MODEL_FILENAME)
        except Exception:
            pass
    # Limpia la caché interna de Streamlit
    st.cache_resource.clear()
    st.error(
        f"Error al cargar el archivo de modelo: {e}. "
        "El archivo corrupto se ha eliminado. Por favor vuelve a cargar la página."
    )
    st.stop()

def preprocess_image(image):
    """Preprocesa la imagen para los formatos requeridos por el modelo."""
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    tensor_features = np.expand_dims(img_array / 255.0, axis=0)
    flat_features = img_array.flatten().reshape(1, -1)
    
    return tensor_features, flat_features

# Formulario de carga de imágenes
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
