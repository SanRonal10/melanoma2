import os
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download

# Configuración de página
st.set_page_config(
    page_title="Detección de Melanoma",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diagnóstico Prematuro de Melanoma")
st.write("Sube una imagen de una lesión cutánea para evaluar si es benigna o melanoma.")

@st.cache_resource
def load_model_from_hf():
    """Descarga el modelo binario de Hugging Face y lo carga de manera segura."""
    # Sustituye con tu repo y nombre de archivo exactos
    model_path = hf_hub_download(
        repo_id="SanRonal10/melanoma-model",  # Cambiar por tu repositorio de Hugging Face
        filename="mimodelo.pkl"
    )
    return joblib.load(model_path)

try:
    model = load_model_from_hf()
except Exception as e:
    st.error(f"Error al cargar el archivo de modelo: {e}. Por favor vuelve a cargar la página.")
    st.stop()

def preprocess_image(image):
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
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
