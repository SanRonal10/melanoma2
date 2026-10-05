import os
import requests
import joblib
import numpy as np
import streamlit as st
from PIL import Image

# Configuración inicial de la interfaz de Streamlit
st.set_page_config(
    page_title="Detección de Melanoma",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diagnóstico Prematuro de Melanoma")
st.write("Sube una imagen de una lesión cutánea para evaluar si es benigna o melanoma.")

# ==============================================================================
# CONFIGURACIÓN DE GOOGLE DRIVE
# ==============================================================================
DRIVE_FILE_ID = "1DOc2I8MRnelWOnssNkAGrSNd4zAkh5m8"
MODEL_FILENAME = "mimodelo.pkl"
# ==============================================================================

def download_file_from_google_drive(file_id, destination):
    """
    Descarga directa desde Google Drive usando requests para evitar errores de '_io.BytesIO'.
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


@st.cache_resource
def load_model():
    """
    Descarga el modelo desde Google Drive si no existe localmente y lo carga con joblib/keras.
    """
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Descargando modelo desde Google Drive (esto solo ocurre una vez)..."):
            download_file_from_google_drive(DRIVE_FILE_ID, MODEL_FILENAME)
            
    # Intentar cargar con Keras o Joblib/Pickle
    try:
        import keras
        return keras.models.load_model(MODEL_FILENAME)
    except Exception:
        return joblib.load(MODEL_FILENAME)


# Intentar cargar el modelo con control de excepciones y limpieza automática
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
    """
    Preprocesa la imagen tanto para formato Tensor 4D (Keras/CNN) como Vector 2D (Scikit-Learn).
    """
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    tensor_features = np.expand_dims(img_array / 255.0, axis=0)
    flat_features = img_array.flatten().reshape(1, -1)
    
    return tensor_features, flat_features


# Carga de la imagen por parte del usuario
uploaded_file = st.file_uploader("Carga una imagen (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)
    
    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Procesando imagen y evaluando..."):
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
                st.error("⚠️ **Resultado:** Posible Melanoma detectado.")
            else:
                st.success("✅ **Resultado:** Posible Lesión Benigna.")
