import os
import requests
import joblib
import numpy as np
import streamlit as st
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Detección de Melanoma",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diagnóstico Prematuro de Melanoma")
st.write("Sube una imagen de una lesión cutánea para evaluar si es benigna o melanoma.")

# URL directa del release en GitHub
MODEL_URL = "https://github.com/SanRonal10/melanoma2/releases/download/v1.0.0/mimodelo.pkl"
LOCAL_MODEL_PATH = "mimodelo_local.pkl"

@st.cache_resource
def load_model_safely():
    """Descarga el modelo a un archivo físico y lo carga con joblib desde disco."""
    # Comprobar si el archivo local existe y tiene un tamaño válido (mayor a 10 KB)
    if not os.path.exists(LOCAL_MODEL_PATH) or os.path.getsize(LOCAL_MODEL_PATH) < 10000:
        if os.path.exists(LOCAL_MODEL_PATH):
            os.remove(LOCAL_MODEL_PATH)
            
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(MODEL_URL, headers=headers, allow_redirects=True, stream=True)
        
        if response.status_code == 200:
            # Escribir directamente en el disco como bytes en lugar de usar BytesIO
            with open(LOCAL_MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            raise RuntimeError(f"Error al descargar el modelo desde GitHub. Código HTTP: {response.status_code}")

    # Cargar pasando el string de la ruta del archivo físico, NO un objeto en memoria
    return joblib.load(LOCAL_MODEL_PATH)

# Carga del modelo
try:
    model = load_model_safely()
except Exception as e:
    # Si falla la carga, eliminar el archivo corrupto para evitar bloqueos
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            os.remove(LOCAL_MODEL_PATH)
        except Exception:
            pass
    st.error(f"Error al cargar el archivo de modelo: {e}. Por favor vuelve a cargar la página.")
    st.stop()

def preprocess_image(image):
    """Ajusta las dimensiones de la imagen y normaliza sus valores."""
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    tensor_features = np.expand_dims(img_array / 255.0, axis=0)
    flat_features = img_array.flatten().reshape(1, -1)
    
    return tensor_features, flat_features

# Interfaz para subir la imagen
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
