import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_instrumentos_multilabel.h5"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Bisturí n.º 4",
    "Pinza de disección recta",
    "Tijera Mayo recta",
    "Tijera Mayo curva",
]

# Umbrales optimizados en validación en el Cuaderno 2.
THRESHOLDS = np.array([0.36, 0.35, 0.47, 0.50], dtype=np.float32)


st.set_page_config(
    page_title="Instrumentos quirúrgicos",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 2rem;}
    .titulo-principal {text-align:center; margin-bottom:0.2rem;}
    .subtitulo {text-align:center; color:#6f625c; margin-bottom:1.5rem;}
    .tarjeta {
        padding: 1rem 1.1rem;
        border: 1px solid #eadfd8;
        border-radius: 14px;
        background: #fffaf7;
        margin-bottom: 0.8rem;
    }
    .detectado {font-weight:700; color:#177245;}
    .no-detectado {font-weight:600; color:#767676;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando el modelo de inteligencia artificial...")
def cargar_modelo():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el archivo modelo_instrumentos_multilabel.h5."
        )

    modelo = tf.keras.models.load_model(MODEL_PATH, compile=False)

    if tuple(modelo.input_shape[1:]) != (*IMG_SIZE, 3):
        raise ValueError(
            f"El modelo espera una entrada {modelo.input_shape}; "
            f"se esperaba (None, {IMG_SIZE[0]}, {IMG_SIZE[1]}, 3)."
        )

    if int(modelo.output_shape[-1]) != len(CLASS_NAMES):
        raise ValueError(
            f"El modelo tiene {modelo.output_shape[-1]} salidas; "
            f"se esperaban {len(CLASS_NAMES)}."
        )

    return modelo


def preparar_imagen(imagen: Image.Image) -> tuple[Image.Image, np.ndarray]:
    imagen_rgb = ImageOps.exif_transpose(imagen).convert("RGB")
    imagen_redimensionada = imagen_rgb.resize(
        IMG_SIZE[::-1], Image.Resampling.BILINEAR
    )

    # El modelo contiene una capa Rescaling interna.
    arreglo = np.asarray(imagen_redimensionada, dtype=np.float32)
    lote = np.expand_dims(arreglo, axis=0)
    return imagen_rgb, lote


def predecir(modelo, imagen: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    _, lote = preparar_imagen(imagen)
    probabilidades = np.asarray(
        modelo.predict(lote, verbose=0)[0], dtype=np.float32
    )
    detectados = probabilidades >= THRESHOLDS
    return probabilidades, detectados


st.markdown(
    '<h1 class="titulo-principal">Reconocimiento multilabel de instrumentos quirúrgicos</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitulo">Prototipo académico basado en MobileNetV2 y transferencia de aprendizaje</p>',
    unsafe_allow_html=True,
)

st.info(
    "El sistema analiza la imagen completa y puede reconocer varias clases "
    "simultáneamente. No localiza los instrumentos mediante cajas."
)

try:
    modelo = cargar_modelo()
except Exception as error:
    st.error("La aplicación no pudo cargar el modelo.")
    st.exception(error)
    st.stop()

pestana_archivo, pestana_camara = st.tabs(["Subir imagen", "Usar cámara"])

with pestana_archivo:
    archivo = st.file_uploader(
        "Selecciona una fotografía",
        type=["jpg", "jpeg", "png", "webp"],
        help="Usa una imagen clara que contenga uno o varios instrumentos.",
    )

with pestana_camara:
    foto = st.camera_input("Toma una fotografía")

entrada = archivo if archivo is not None else foto

if entrada is None:
    st.markdown(
        """
        <div class="tarjeta">
        <b>Instrucciones:</b><br>
        1. Sube una imagen o utiliza la cámara.<br>
        2. Presiona <b>Analizar imagen</b>.<br>
        3. Revisa las probabilidades y las clases detectadas.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    imagen = Image.open(entrada)
    col_imagen, col_resultados = st.columns([1, 1.15], gap="large")

    with col_imagen:
        st.subheader("Imagen de entrada")
        st.image(imagen, use_container_width=True)

    with col_resultados:
        st.subheader("Predicción")
        analizar = st.button(
            "Analizar imagen",
            type="primary",
            use_container_width=True,
        )

        if analizar:
            with st.spinner("Analizando la imagen..."):
                probabilidades, detectados = predecir(modelo, imagen)

            indices_detectados = np.where(detectados)[0]
            if len(indices_detectados) > 0:
                st.success(
                    f"Se reconocieron {len(indices_detectados)} clase(s) de instrumento."
                )
                for indice in indices_detectados:
                    st.markdown(
                        f'<div class="tarjeta"><span class="detectado">✓ '
                        f'{CLASS_NAMES[indice]}</span><br>'
                        f'Confianza: <b>{probabilidades[indice]:.2%}</b></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("Ninguna clase superó su umbral de decisión.")

            resultados = pd.DataFrame(
                {
                    "Instrumento": CLASS_NAMES,
                    "Probabilidad": probabilidades,
                    "Umbral": THRESHOLDS,
                    "Resultado": [
                        "Detectado" if valor else "No detectado"
                        for valor in detectados
                    ],
                }
            ).sort_values("Probabilidad", ascending=False)

            st.subheader("Resultados por instrumento")
            st.dataframe(
                resultados.style.format(
                    {
                        "Probabilidad": "{:.2%}",
                        "Umbral": "{:.2f}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

            grafica = resultados.set_index("Instrumento")[["Probabilidad", "Umbral"]]
            st.subheader("Probabilidades y umbrales")
            st.bar_chart(grafica, horizontal=True)

st.divider()
st.caption(
    "Prototipo académico. No sustituye el conteo protocolizado ni la "
    "verificación realizada por profesionales de salud."
)
