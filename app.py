import os
import urllib.request
import joblib
import numpy as np
import streamlit as st
from PIL import Image

# Configuración inicial de Streamlit
st.set_page_config(
    page_title="Detección de Melanoma",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diagnóstico Prematuro de Melanoma")
st.write("Sube una imagen de una lesión cutánea para evaluar si es benigna o melanoma.")

# ID de Google Drive y nombre local del modelo
DRIVE_FILE_ID = "1DOc2I8MRnelWOnssNkAGrSNd4zAkh5m8"
MODEL_FILENAME = "mimodelo.pkl"

@st.cache_resource
def load_model():
    """Descarga el modelo binario de Google Drive si no existe localmente."""
    if not os.path.exists(MODEL_FILENAME):
        # Enlace con confirmación directa para evitar que Drive devuelva una página HTML
        url = f"https://drive.google.com/uc?export=download&confirm=t&id={DRIVE_FILE_ID}"
        
        with st.spinner("Descargando archivo de modelo desde Google Drive..."):
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req) as response, open(MODEL_FILENAME, 'wb') as out_file:
                out_file.write(response.read())

        # Validación: si el archivo es un HTML (menor a 100 KB o contiene <html>), lanzamos un error
        if os.path.getsize(MODEL_FILENAME) < 1024 * 100:
            with open(MODEL_FILENAME, "r", errors="ignore") as f:
                content = f.read()
                if "<html" in content.lower():
                    os.remove(MODEL_FILENAME)
                    raise ValueError("Google Drive devolvió una página HTML en lugar del archivo binario .pkl. Verifica el acceso público del archivo.")

    # Intentar des-serializar con joblib
    return joblib.load(MODEL_FILENAME)

# Carga y verificación del modelo
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
    """Preprocesa la imagen para compatibilidad con modelos 2D o 4D."""
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    tensor_features = np.expand_dims(img_array / 255.0, axis=0)
    flat_features = img_array.flatten().reshape(1, -1)
    
    return tensor_features, flat_features

# Carga de la imagen por el usuario
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
