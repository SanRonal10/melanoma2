import os
import dill
import gdown
import joblib
import numpy as np
import streamlit as st
from PIL import Image

# Importar TensorFlow y Keras
import tensorflow as tf
import keras

# Configuración de interfaz
st.set_page_config(
    page_title="Detección de Melanoma", page_icon="🩺", layout="centered"
)

st.title("🩺 Diagnóstico Prematuro de Melanoma")
st.write(
    "Sube una imagen de una lesión cutánea para evaluar si es benigna o melanoma."
)

# ==============================================================================
# CONFIGURACIÓN DE GOOGLE DRIVE
# ==============================================================================
DRIVE_FILE_ID = "1DOc2I8MRnelWOnssNkAGrSNd4zAkh5m8"
MODEL_FILENAME = "mimodelo.pkl"
# ==============================================================================


@st.cache_resource
def load_model():
    """Descarga el modelo desde Google Drive e intenta cargarlo como

    Keras o Pickle/Joblib según su estructura.
    """
    if not os.path.exists(MODEL_FILENAME):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        with st.spinner(
            "Descargando modelo desde Google Drive (solo ocurre la primera vez)..."
        ):
            gdown.download(url=url, output=MODEL_FILENAME, quiet=False)

    # Intento 1: Carga nativa de Keras (si el .pkl es un modelo Keras re-nombrado o exportado)
    try:
        return keras.models.load_model(MODEL_FILENAME)
    except Exception:
        pass

    # Intento 2: Carga estándar de Joblib/Pickle/Dill
    return joblib.load(MODEL_FILENAME)


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
    """Preprocesa la imagen tanto en formato Tensor 4D (Keras/CNN)

    como en Vector 2D (Scikit-Learn).
    """
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)

    # Tensor 4D normalizado [0, 1] para Keras / TensorFlow
    tensor_features = np.expand_dims(img_array / 255.0, axis=0)

    # Vector aplanado 2D para Scikit-Learn
    flat_features = img_array.flatten().reshape(1, -1)

    return tensor_features, flat_features


uploaded_file = st.file_uploader(
    "Carga una imagen (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)

    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Procesando imagen y evaluando..."):
            tensor_img, flat_img = preprocess_image(image)

            # Intentar predicción con formato Tensor 4D primero (Keras)
            try:
                prediction = model.predict(tensor_img)
            except Exception:
                prediction = model.predict(flat_img)

            # Extracción del resultado numérico
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
