import os
import gdown
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
# Coloca aquí el FILE_ID de tu archivo subido a Google Drive
DRIVE_FILE_ID = "1DOc2I8MRnelWOnssNkAGrSNd4zAkh5m8"
MODEL_FILENAME = "mimodelo.pkl"
# ==============================================================================

@st.cache_resource
def load_model():
    """
    Descarga el modelo desde Google Drive si no existe localmente
    y lo carga en memoria con joblib.
    """
    if not os.path.exists(MODEL_FILENAME):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        with st.spinner("Descargando modelo desde Google Drive..."):
            # fuzzy=True permite omitir la pantalla de confirmación de virus en archivos grandes
            gdown.download(url, MODEL_FILENAME, quiet=False, fuzzy=True)
            
    return joblib.load(MODEL_FILENAME)


# Intentar cargar el modelo con control de excepciones
try:
    model = load_model()
except Exception as e:
    # Si el archivo descargado está corrupto o es HTML, se elimina para no bloquear futuros intentos
    if os.path.exists(MODEL_FILENAME):
        os.remove(MODEL_FILENAME)
    st.error(
        f"Error al cargar el archivo de modelo: {e}. "
        "El archivo corrupto se ha eliminado. Por favor vuelve a cargar la página."
    )
    st.stop()


def preprocess_image(image):
    """
    Preprocesa la imagen cargada tanto para modelos tabulares/scikit-learn (2D)
    como para redes neuronales/Keras/TensorFlow (4D).
    """
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    # Formato aplanado 2D (Scikit-Learn, XGBoost, Random Forest)
    flat_features = img_array.flatten().reshape(1, -1)
    
    # Formato tensor 4D normalizado (TensorFlow, Keras, CNNs)
    tensor_features = np.expand_dims(img_array / 255.0, axis=0)
    
    return flat_features, tensor_features


# Carga de la imagen por parte del usuario
uploaded_file = st.file_uploader("Carga una imagen (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)
    
    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Procesando imagen y evaluando..."):
            flat_img, tensor_img = preprocess_image(image)
            
            # Intento híbrido de predicción según la estructura del modelo
            try:
                prediction = model.predict(flat_img)
            except Exception:
                prediction = model.predict(tensor_img)
            
            # Extracción del valor numérico de la predicción
            if isinstance(prediction, (list, np.ndarray)):
                pred_val = prediction[0]
                if isinstance(pred_val, (list, np.ndarray)):
                    pred_val = pred_val[0]
            else:
                pred_val = prediction
                
            pred_num = float(pred_val)
            
            # Umbral de diagnóstico
            if pred_num >= 0.5 or pred_num == 1:
                st.error("⚠️ **Resultado:** Posible Melanoma detectado.")
            else:
                st.success("✅ **Resultado:** Posible Lesión Benigna.")
