from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import re
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

APP_DIR = Path(__file__).resolve().parent

CLASS_NAMES = [
    "Bisturí n.º 4",
    "Pinza de disección recta",
    "Tijera Mayo recta",
    "Tijera Mayo curva",
]

THRESHOLDS = np.array(
    [0.57, 0.70, 0.59, 0.48],
    dtype=np.float32,
)

IMG_SIZE = (224, 224)
NUM_CLASSES = len(CLASS_NAMES)

ABREVIATURAS = {
    "Bisturí n.º 4": "B4",
    "Pinza de disección recta": "PR",
    "Tijera Mayo recta": "MR",
    "Tijera Mayo curva": "MC",
}

MODEL_CANDIDATES = [
    APP_DIR / "modelo_instrumentos_multilabel.h5",
    APP_DIR / "4_modelo_instrumentos_multilabel(2).h5",
]

MODEL_PATH = next(
    (ruta for ruta in MODEL_CANDIDATES if ruta.exists()),
    MODEL_CANDIDATES[0],
)

FEEDBACK_DIR = Path(
    os.getenv(
        "SURGIVISION_FEEDBACK_DIR",
        str(APP_DIR / "data_feedback"),
    )
).expanduser()

IMAGES_DIR = FEEDBACK_DIR / "imagenes"
CSV_PATH = FEEDBACK_DIR / "feedback_validado.csv"
JSONL_PATH = FEEDBACK_DIR / "feedback_validado.jsonl"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

PREDICT_LOCK = threading.Lock()
WRITE_LOCK = threading.Lock()


st.set_page_config(
    page_title="SurgiVision AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# ESTILOS
# =============================================================================

CSS = """
<style>
:root {
    --sv-navy-950: #061a30;
    --sv-navy-900: #0b2748;
    --sv-blue-650: #0e6fa3;
    --sv-blue-100: #eaf6fc;
    --sv-teal-750: #0b6b63;
    --sv-teal-100: #dff7f2;
    --sv-green-750: #067647;
    --sv-green-100: #ecfdf3;
    --sv-amber-750: #b54708;
    --sv-amber-100: #fff7e8;
    --sv-red-750: #b42318;
    --sv-red-100: #fff1f0;
    --sv-violet-750: #6941c6;
    --sv-violet-100: #f4f0ff;
    --sv-slate-950: #102a43;
    --sv-slate-750: #486581;
    --sv-slate-500: #829ab1;
    --sv-slate-300: #d9e2ec;
    --sv-slate-100: #f0f4f8;
    --sv-surface: #ffffff;
    --sv-background: #f3f7fa;
}

.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(14,111,163,.07), transparent 22%),
        var(--sv-background);
}

.block-container {
    max-width: 1450px;
    padding-top: 1.1rem;
    padding-bottom: 2.5rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fbfd 0%, #eef4f8 100%);
    border-right: 1px solid var(--sv-slate-300);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.4rem;
}

.sv-hero {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 89% 8%, rgba(255,255,255,.20), transparent 23%),
        radial-gradient(circle at 72% 95%, rgba(87,221,201,.16), transparent 26%),
        linear-gradient(122deg, var(--sv-navy-950) 0%, var(--sv-blue-650) 62%, var(--sv-teal-750) 100%);
    border-radius: 26px;
    padding: 31px 35px;
    margin-bottom: 14px;
    color: white;
    box-shadow: 0 20px 48px rgba(7,28,51,.20);
}

.sv-hero-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
}

.sv-brand {
    display: flex;
    align-items: center;
    gap: 17px;
}

.sv-brand-mark {
    width: 64px;
    height: 64px;
    min-width: 64px;
    display: grid;
    place-items: center;
    border-radius: 19px;
    background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.30);
    font-size: 22px;
    font-weight: 900;
}

.sv-title {
    margin: 0;
    color: white;
    font-size: 32px;
    line-height: 1.07;
    letter-spacing: -.65px;
}

.sv-subtitle {
    margin: 7px 0 0;
    color: rgba(255,255,255,.88);
    font-size: 15px;
}

.sv-ready {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 10px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.25);
    font-size: 12px;
    font-weight: 800;
    white-space: nowrap;
}

.sv-ready-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #6ce9a6;
    box-shadow: 0 0 0 4px rgba(108,233,166,.18);
}

.sv-prototype {
    margin-top: 18px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .85px;
    color: rgba(255,255,255,.72);
}

.sv-feature-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0,1fr));
    gap: 10px;
    margin-bottom: 15px;
}

.sv-feature-card {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 12px 14px;
    background: rgba(255,255,255,.94);
    border: 1px solid var(--sv-slate-300);
    border-radius: 15px;
    box-shadow: 0 6px 16px rgba(31,55,76,.05);
}

.sv-feature-icon {
    width: 32px;
    height: 32px;
    min-width: 32px;
    display: grid;
    place-items: center;
    border-radius: 10px;
    background: var(--sv-blue-100);
    color: var(--sv-blue-650);
    font-size: 15px;
    font-weight: 900;
}

.sv-feature-title {
    color: var(--sv-slate-950);
    font-size: 12px;
    font-weight: 850;
}

.sv-feature-copy {
    margin-top: 2px;
    color: var(--sv-slate-750);
    font-size: 11px;
}

.sv-section-kicker {
    color: var(--sv-teal-750);
    font-size: 11px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 1.25px;
    margin-bottom: 4px;
}

.sv-section-title {
    color: var(--sv-navy-900);
    font-size: 22px;
    font-weight: 850;
    margin: 0 0 6px;
    letter-spacing: -.25px;
}

.sv-section-copy {
    color: var(--sv-slate-750);
    font-size: 14px;
    line-height: 1.5;
    margin-bottom: 14px;
}

.sv-panel {
    background: rgba(255,255,255,.97);
    border: 1px solid var(--sv-slate-300);
    border-radius: 22px;
    padding: 21px;
    box-shadow: 0 10px 30px rgba(31,55,76,.075);
    margin-bottom: 14px;
}

.sv-class-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0,1fr));
    gap: 10px;
    margin: 12px 0 4px;
}

.sv-class-card {
    display: flex;
    align-items: center;
    gap: 12px;
    min-height: 72px;
    padding: 13px 14px;
    border: 1px solid var(--sv-slate-300);
    border-radius: 15px;
    background: linear-gradient(145deg, #fff 0%, #f8fcfd 100%);
}

.sv-class-badge {
    width: 39px;
    height: 39px;
    min-width: 39px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    background: var(--sv-teal-100);
    color: var(--sv-teal-750);
    font-weight: 900;
    font-size: 12px;
}

.sv-class-name {
    color: var(--sv-slate-950);
    font-size: 13px;
    font-weight: 800;
    line-height: 1.25;
}

.sv-class-caption {
    margin-top: 4px;
    color: var(--sv-slate-500);
    font-size: 11px;
}

.sv-note {
    margin-top: 11px;
    padding: 11px 13px;
    border-radius: 12px;
    background: var(--sv-slate-100);
    color: var(--sv-slate-750);
    font-size: 12px;
    line-height: 1.48;
}

.sv-status {
    border-radius: 18px;
    padding: 20px 21px;
    border: 1px solid;
    margin-bottom: 13px;
    box-shadow: 0 6px 16px rgba(31,55,76,.035);
}

.sv-status-idle {
    background: linear-gradient(145deg, #eff6ff, #f8fbff);
    border-color: #b2ddff;
    color: #1849a9;
}

.sv-status-success {
    background: linear-gradient(145deg, var(--sv-green-100), #f7fff9);
    border-color: #abefc6;
    color: var(--sv-green-750);
}

.sv-status-empty {
    background: linear-gradient(145deg, #f8fafc, #fff);
    border-color: #cbd5e1;
    color: #334155;
}

.sv-status-warning {
    background: linear-gradient(145deg, var(--sv-amber-100), #fffdf7);
    border-color: #fedf89;
    color: var(--sv-amber-750);
}

.sv-status-error {
    background: linear-gradient(145deg, var(--sv-red-100), #fff8f7);
    border-color: #fecdca;
    color: var(--sv-red-750);
}

.sv-status-title {
    margin: 0 0 7px;
    font-size: 18px;
    font-weight: 900;
}

.sv-status-copy {
    margin: 0;
    font-size: 14px;
    line-height: 1.55;
}

.sv-result-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0,1fr));
    gap: 9px;
    margin-top: 13px;
}

.sv-result-card {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 13px;
    border-radius: 13px;
    background: white;
    border: 1px solid #abefc6;
    color: #05603a;
    font-size: 13px;
    font-weight: 820;
}

.sv-result-check {
    width: 25px;
    height: 25px;
    min-width: 25px;
    display: grid;
    place-items: center;
    border-radius: 50%;
    background: #d1fadf;
    font-weight: 950;
}

.sv-summary-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0,1fr));
    gap: 10px;
    margin-bottom: 12px;
}

.sv-summary-card {
    min-height: 105px;
    padding: 14px;
    border: 1px solid var(--sv-slate-300);
    border-radius: 15px;
    background: linear-gradient(155deg, white, #fbfdfe);
}

.sv-summary-label {
    color: var(--sv-slate-500);
    font-size: 11px;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: .58px;
}

.sv-summary-value {
    margin-top: 7px;
    color: var(--sv-navy-900);
    font-size: 18px;
    font-weight: 900;
    line-height: 1.15;
}

.sv-summary-note {
    margin-top: 5px;
    color: var(--sv-slate-750);
    font-size: 11px;
    line-height: 1.38;
}

.sv-context {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    padding: 12px 14px;
    border-radius: 13px;
    background: var(--sv-slate-100);
    border: 1px solid var(--sv-slate-300);
    color: var(--sv-slate-750);
    font-size: 12px;
}

.sv-feedback {
    padding: 16px;
    border-radius: 17px;
    background: linear-gradient(145deg, rgba(244,240,255,.88), rgba(255,255,255,.96));
    border: 1px solid #d9d6fe;
    box-shadow: 0 7px 20px rgba(105,65,198,.055);
    margin-top: 14px;
}

.sv-feedback-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 8px;
}

.sv-feedback-title {
    color: #42307d;
    font-size: 15px;
    font-weight: 900;
}

.sv-feedback-badge {
    padding: 5px 9px;
    border-radius: 999px;
    background: var(--sv-violet-100);
    color: var(--sv-violet-750);
    border: 1px solid #d9d6fe;
    font-size: 10px;
    font-weight: 850;
}

.sv-feedback-copy {
    color: #594e72;
    font-size: 12px;
    line-height: 1.5;
    margin-bottom: 10px;
}

.sv-footer {
    margin-top: 17px;
    padding: 14px 16px;
    border-radius: 14px;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #9a3412;
    font-size: 12px;
    line-height: 1.55;
}

div.stButton > button {
    min-height: 46px;
    border-radius: 12px;
    font-weight: 780;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,.97);
    border: 1px solid var(--sv-slate-300);
    border-radius: 22px;
    box-shadow: 0 10px 30px rgba(31,55,76,.075);
}

[data-testid="stFileUploaderDropzone"],
[data-testid="stCameraInput"] {
    border-radius: 16px;
}

@media (max-width: 900px) {
    .sv-hero-row {
        align-items: flex-start;
        flex-direction: column;
    }
    .sv-feature-grid,
    .sv-summary-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 720px) {
    .sv-hero {
        padding: 23px;
    }
    .sv-title {
        font-size: 25px;
    }
    .sv-class-grid,
    .sv-result-grid,
    .sv-context {
        grid-template-columns: 1fr;
    }
}
</style>
"""


# =============================================================================
# MODELO E INFERENCIA
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str) -> tf.keras.Model:
    """Carga una única instancia compartida del modelo."""

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            "No se encontró modelo_instrumentos_multilabel.h5 "
            "en la carpeta de la aplicación."
        )

    model = tf.keras.models.load_model(
        str(path),
        compile=False,
    )

    input_shape = tuple(model.input_shape[1:])
    output_units = int(model.output_shape[-1])

    if input_shape != (*IMG_SIZE, 3):
        raise ValueError(
            f"El modelo espera una entrada {model.input_shape}; "
            f"la aplicación requiere (None, {IMG_SIZE[0]}, {IMG_SIZE[1]}, 3)."
        )

    if output_units != NUM_CLASSES:
        raise ValueError(
            f"El modelo tiene {output_units} salidas; se esperaban {NUM_CLASSES}."
        )

    return model


def convert_to_pil(source: Any) -> Image.Image:
    """Convierte una ruta, imagen PIL, archivo o arreglo en una imagen RGB."""

    if isinstance(source, Image.Image):
        image = source.copy()
    elif isinstance(source, (str, Path)):
        image = Image.open(source)
    elif hasattr(source, "read"):
        image = Image.open(source)
    else:
        array = np.asarray(source)
        if array.ndim not in (2, 3):
            raise ValueError("La entrada no corresponde a una imagen válida.")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)

    return ImageOps.exif_transpose(image).convert("RGB")


def generate_views(
    image: Image.Image,
    crop_ratio: float = 0.85,
) -> dict[str, Image.Image]:
    """Genera la vista original, reflejada y recortada."""

    if not 0 < crop_ratio <= 1:
        raise ValueError("La proporción de recorte debe estar entre 0 y 1.")

    width, height = image.size
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2

    central_crop = image.crop(
        (left, top, left + crop_width, top + crop_height)
    )

    return {
        "Original": image,
        "Flip horizontal": ImageOps.mirror(image),
        "Recorte central": central_crop,
    }


def predict_instruments(
    image: Image.Image,
    model: tf.keras.Model,
) -> dict[str, Any]:
    """
    Ejecuta la misma inferencia utilizada en el cuaderno final.

    La probabilidad final es el máximo entre tres vistas TTA y se compara
    con el umbral específico de cada clase.
    """

    image = convert_to_pil(image)

    if image.width < 20 or image.height < 20:
        raise ValueError(
            "La imagen es demasiado pequeña. Utilice una imagen "
            "de al menos 20 × 20 píxeles."
        )

    views = generate_views(image)

    batch = np.stack(
        [
            np.asarray(
                view.resize(
                    IMG_SIZE[::-1],
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            )
            for view in views.values()
        ],
        axis=0,
    )

    # El modelo ya contiene una capa Rescaling.
    # No se divide nuevamente entre 255.
    with PREDICT_LOCK:
        view_probabilities = np.asarray(
            model.predict(batch, verbose=0),
            dtype=np.float32,
        )

    if view_probabilities.shape != (3, NUM_CLASSES):
        raise RuntimeError(
            f"Se esperaba una matriz (3, {NUM_CLASSES}) y se obtuvo "
            f"{view_probabilities.shape}."
        )

    final_probabilities = view_probabilities.max(axis=0)
    detected_mask = final_probabilities >= THRESHOLDS
    agreement = (
        (view_probabilities >= THRESHOLDS).mean(axis=0) * 100
    )

    table = pd.DataFrame(
        {
            "Instrumento": CLASS_NAMES,
            "Original": view_probabilities[0],
            "Flip horizontal": view_probabilities[1],
            "Recorte central": view_probabilities[2],
            "Probabilidad final": final_probabilities,
            "Umbral": THRESHOLDS,
            "Margen": final_probabilities - THRESHOLDS,
            "Detectado": detected_mask,
            "Acuerdo de vistas": agreement,
        }
    ).sort_values(
        "Probabilidad final",
        ascending=False,
    ).reset_index(drop=True)

    detected = table.loc[
        table["Detectado"],
        "Instrumento",
    ].tolist()

    best_row = table.iloc[0]

    return {
        "imagen": image,
        "tabla": table,
        "instrumentos_detectados": detected,
        "hay_detecciones": bool(detected),
        "estado": "DETECCION" if detected else "SIN_DETECCIONES",
        "mensaje": (
            f"Se detectó {len(detected)} instrumento."
            if len(detected) == 1
            else (
                f"Se detectaron {len(detected)} instrumentos."
                if detected
                else (
                    "No se encontraron instrumentos reconocidos dentro "
                    "de los umbrales configurados."
                )
            )
        ),
        "mejor_candidato": str(best_row["Instrumento"]),
        "probabilidad_maxima": float(
            best_row["Probabilidad final"]
        ),
        "umbral_mejor_candidato": float(best_row["Umbral"]),
        "probabilidades_vistas": view_probabilities,
    }


# =============================================================================
# ALMACENAMIENTO DE VALIDACIONES
# =============================================================================

CSV_FIELDS = [
    "id_registro",
    "fecha_validacion_utc",
    "archivo_imagen",
    "sha256_imagen",
    "codigo_control",
    "momento_control",
    "modelo_version",
    "estado_modelo",
    "etiquetas_predichas",
    "etiquetas_validadas",
    "vector_etiquetas",
    "tipo_feedback",
    "probabilidades",
    "observaciones",
]


def sanitize_fragment(text: str) -> str:
    value = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(text or "").strip(),
    )
    return value.strip("_")[:36] or "sin_codigo"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def count_feedback_rows() -> int:
    if not CSV_PATH.exists():
        return 0

    with CSV_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return max(sum(1 for _ in file) - 1, 0)


def extract_probabilities(
    result: dict[str, Any],
) -> dict[str, float]:
    table = result.get("tabla")
    if table is None:
        return {}

    return {
        str(row["Instrumento"]): float(
            row["Probabilidad final"]
        )
        for _, row in table.iterrows()
    }


def append_csv(record: dict[str, Any]) -> None:
    exists = CSV_PATH.exists()
    row = dict(record)

    for field in (
        "etiquetas_predichas",
        "etiquetas_validadas",
        "vector_etiquetas",
        "probabilidades",
    ):
        row[field] = json.dumps(
            row[field],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    with CSV_PATH.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                field: row.get(field, "")
                for field in CSV_FIELDS
            }
        )


def append_jsonl(record: dict[str, Any]) -> None:
    with JSONL_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def save_validation(
    result: dict[str, Any],
    validated_labels: list[str],
    code: str,
    moment: str,
    observations: str,
) -> dict[str, Any]:
    """Guarda imagen, predicción y etiquetas validadas."""

    predicted_labels = [
        label
        for label in result["instrumentos_detectados"]
        if label in CLASS_NAMES
    ]
    validated_labels = [
        label
        for label in validated_labels
        if label in CLASS_NAMES
    ]

    feedback_type = (
        "confirmacion"
        if set(validated_labels) == set(predicted_labels)
        else "correccion"
    )

    date = datetime.now(timezone.utc)
    record_id = uuid.uuid4().hex
    filename = (
        f"{date.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{sanitize_fragment(code)}_"
        f"{record_id[:10]}.jpg"
    )
    image_path = IMAGES_DIR / filename

    with WRITE_LOCK:
        result["imagen"].convert("RGB").save(
            image_path,
            format="JPEG",
            quality=95,
            optimize=True,
        )

        vector = [
            int(label in validated_labels)
            for label in CLASS_NAMES
        ]

        record = {
            "id_registro": record_id,
            "fecha_validacion_utc": date.isoformat(),
            "archivo_imagen": str(image_path),
            "sha256_imagen": sha256_file(image_path),
            "codigo_control": str(code or "").strip(),
            "momento_control": str(moment or ""),
            "modelo_version": MODEL_PATH.name,
            "estado_modelo": result["estado"],
            "etiquetas_predichas": predicted_labels,
            "etiquetas_validadas": validated_labels,
            "vector_etiquetas": vector,
            "tipo_feedback": feedback_type,
            "probabilidades": extract_probabilities(result),
            "observaciones": str(observations or "").strip(),
        }

        append_csv(record)
        append_jsonl(record)

    return record


def build_feedback_zip() -> bytes:
    """Construye un ZIP descargable con CSV, JSONL e imágenes."""

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        if CSV_PATH.exists():
            archive.write(
                CSV_PATH,
                arcname=CSV_PATH.name,
            )

        if JSONL_PATH.exists():
            archive.write(
                JSONL_PATH,
                arcname=JSONL_PATH.name,
            )

        if IMAGES_DIR.exists():
            for image_path in sorted(
                IMAGES_DIR.glob("*")
            ):
                if image_path.is_file():
                    archive.write(
                        image_path,
                        arcname=f"imagenes/{image_path.name}",
                    )

    return buffer.getvalue()


# =============================================================================
# HTML DE LA INTERFAZ
# =============================================================================

def class_cards_html() -> str:
    cards = []

    for name in CLASS_NAMES:
        cards.append(
            f"""
            <div class="sv-class-card">
                <div class="sv-class-badge">
                    {html.escape(ABREVIATURAS[name])}
                </div>
                <div>
                    <div class="sv-class-name">
                        {html.escape(name)}
                    </div>
                    <div class="sv-class-caption">
                        Clase disponible en el modelo
                    </div>
                </div>
            </div>
            """
        )

    return (
        '<div class="sv-class-grid">'
        + "".join(cards)
        + "</div>"
        + """
        <div class="sv-note">
            El sistema reconoce únicamente las clases mostradas.
            Un instrumento diferente puede no ser identificado aunque
            esté presente en la imagen.
        </div>
        """
    )


def result_status_html(
    result: dict[str, Any] | None,
) -> str:
    if result is None:
        return """
        <div class="sv-status sv-status-idle">
            <p class="sv-status-title">Preparado para iniciar</p>
            <p class="sv-status-copy">
                Cargue o capture una imagen y seleccione
                <b>Analizar imagen</b>.
            </p>
        </div>
        """

    detected = result["instrumentos_detectados"]

    if detected:
        cards = "".join(
            f"""
            <div class="sv-result-card">
                <span class="sv-result-check">✓</span>
                <span>{html.escape(label)}</span>
            </div>
            """
            for label in detected
        )

        count_text = (
            "Se reconoció 1 tipo de instrumento."
            if len(detected) == 1
            else (
                f"Se reconocieron {len(detected)} "
                "tipos de instrumentos."
            )
        )

        return f"""
        <div class="sv-status sv-status-success">
            <p class="sv-status-title">
                Instrumentos reconocidos
            </p>
            <p class="sv-status-copy">
                {count_text} Compruebe visualmente la identificación
                antes de guardarla.
            </p>
            <div class="sv-result-grid">{cards}</div>
        </div>
        """

    return """
    <div class="sv-status sv-status-empty">
        <p class="sv-status-title">
            No se reconocieron instrumentos
        </p>
        <p class="sv-status-copy">
            Ninguna de las cuatro clases disponibles fue identificada.
            Confirme el resultado o seleccione las clases correctas
            cuando el modelo haya omitido algún instrumento.
        </p>
    </div>
    """


def summary_html(
    result: dict[str, Any] | None,
    code: str,
    moment: str,
    feedback_saved: bool,
) -> str:
    if result is None:
        result_value = "Pendiente"
        result_note = "Aún no se ha ejecutado el análisis."
        image_value = "Sin analizar"
        image_note = "Cargue una fotografía para continuar."
    else:
        detected = result["instrumentos_detectados"]
        image_value = "Analizada"
        image_note = "El modelo completó el procesamiento."

        if not detected:
            result_value = "Sin reconocimiento"
            result_note = (
                "Ninguna clase disponible fue identificada."
            )
        elif len(detected) == 1:
            result_value = "1 tipo reconocido"
            result_note = detected[0]
        else:
            result_value = (
                f"{len(detected)} tipos reconocidos"
            )
            result_note = "Resultado multilabel."

    validation_value = (
        "Guardada"
        if feedback_saved
        else "Pendiente"
    )
    validation_note = (
        "La revisión se incorporó al conjunto validado."
        if feedback_saved
        else "Confirme o corrija las etiquetas."
    )

    safe_code = html.escape(
        str(code).strip() or "No registrado"
    )
    safe_moment = html.escape(
        str(moment or "No especificado")
    )

    return f"""
    <div class="sv-summary-grid">
        <div class="sv-summary-card">
            <div class="sv-summary-label">Resultado</div>
            <div class="sv-summary-value">
                {html.escape(result_value)}
            </div>
            <div class="sv-summary-note">
                {html.escape(result_note)}
            </div>
        </div>
        <div class="sv-summary-card">
            <div class="sv-summary-label">Imagen</div>
            <div class="sv-summary-value">{image_value}</div>
            <div class="sv-summary-note">{image_note}</div>
        </div>
        <div class="sv-summary-card">
            <div class="sv-summary-label">Validación</div>
            <div class="sv-summary-value">
                {validation_value}
            </div>
            <div class="sv-summary-note">
                {validation_note}
            </div>
        </div>
    </div>
    <div class="sv-context">
        <div><b>Código de control:</b> {safe_code}</div>
        <div><b>Momento:</b> {safe_moment}</div>
    </div>
    """


# =============================================================================
# ESTADO DE SESIÓN
# =============================================================================

SESSION_DEFAULTS = {
    "result": None,
    "analyzed_image": None,
    "source_hash": None,
    "feedback_saved": False,
    "feedback_record": None,
    "feedback_message": "",
    "show_correction": False,
    "corrected_labels": [],
    "widget_nonce": 0,
}


for key, default in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def reset_analysis() -> None:
    for key, default in SESSION_DEFAULTS.items():
        st.session_state[key] = default


# =============================================================================
# APLICACIÓN
# =============================================================================

st.html(CSS)

st.html(
    """
    <div class="sv-hero">
        <div class="sv-hero-row">
            <div class="sv-brand">
                <div class="sv-brand-mark">SV</div>
                <div>
                    <h1 class="sv-title">SurgiVision AI</h1>
                    <p class="sv-subtitle">
                        Asistente visual para reconocimiento de
                        instrumental quirúrgico
                    </p>
                </div>
            </div>
            <div class="sv-ready">
                <span class="sv-ready-dot"></span>
                Sistema disponible
            </div>
        </div>
        <div class="sv-prototype">
            PROTOTIPO ACADÉMICO · APOYO VISUAL CON VALIDACIÓN HUMANA
        </div>
    </div>
    """
)

st.html(
    """
    <div class="sv-feature-grid">
        <div class="sv-feature-card">
            <div class="sv-feature-icon">4</div>
            <div>
                <div class="sv-feature-title">
                    Clases reconocibles
                </div>
                <div class="sv-feature-copy">
                    Instrumental incluido en el entrenamiento.
                </div>
            </div>
        </div>
        <div class="sv-feature-card">
            <div class="sv-feature-icon">✓</div>
            <div>
                <div class="sv-feature-title">
                    Validación humana
                </div>
                <div class="sv-feature-copy">
                    El usuario confirma o corrige el resultado.
                </div>
            </div>
        </div>
        <div class="sv-feature-card">
            <div class="sv-feature-icon">↻</div>
            <div>
                <div class="sv-feature-title">
                    Mejora continua
                </div>
                <div class="sv-feature-copy">
                    Las revisiones pueden apoyar nuevos entrenamientos.
                </div>
            </div>
        </div>
    </div>
    """
)

try:
    model = load_model_cached(str(MODEL_PATH))
except Exception as error:
    st.error(
        "No fue posible cargar el modelo. "
        f"Detalle: {error}"
    )
    st.info(
        "Coloque el archivo "
        "`modelo_instrumentos_multilabel.h5` "
        "en la misma carpeta que `streamlit_app.py`."
    )
    st.stop()


with st.sidebar:
    st.markdown("### Panel de control")
    st.caption(
        "Configure el contexto del análisis y gestione "
        "las validaciones guardadas."
    )

    code = st.text_input(
        "Código de bandeja o procedimiento",
        placeholder="Ej.: BAN-024",
        help="Campo opcional. No incluya datos del paciente.",
        key="control_code",
    )

    moment = st.selectbox(
        "Momento del control",
        [
            "Antes del procedimiento",
            "Durante el procedimiento",
            "Después del procedimiento",
            "Control de bandeja",
            "Demostración académica",
        ],
        key="control_moment",
    )

    st.divider()

    st.markdown("#### Conjunto validado")
    feedback_count = count_feedback_rows()
    st.metric(
        "Registros guardados",
        feedback_count,
    )

    if CSV_PATH.exists():
        st.download_button(
            "Descargar registro CSV",
            data=CSV_PATH.read_bytes(),
            file_name="feedback_validado.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Descargar conjunto completo ZIP",
            data=build_feedback_zip(),
            file_name="surgivision_feedback.zip",
            mime="application/zip",
            use_container_width=True,
        )
    else:
        st.caption(
            "Todavía no existen validaciones para descargar."
        )

    st.warning(
        "En Streamlit Community Cloud, los archivos locales "
        "no tienen persistencia garantizada. Descargue el CSV "
        "o el ZIP antes de reiniciar la aplicación."
    )

    with st.expander("Información del modelo"):
        st.write(f"**Archivo:** `{MODEL_PATH.name}`")
        st.write("**Entrada:** 224 × 224 × 3")
        st.write("**Salida:** 4 probabilidades sigmoide")
        st.write("**Inferencia:** tres vistas TTA")
        st.write("**Decisión:** umbral independiente por clase")


left, right = st.columns(
    [0.88, 1.12],
    gap="large",
)

with left:
    with st.container(border=True):
        st.html(
            """
            <div class="sv-section-kicker">Captura</div>
            <div class="sv-section-title">Registrar imagen</div>
            <div class="sv-section-copy">
                Tome una fotografía centrada, con iluminación
                uniforme y con los instrumentos completamente visibles.
            </div>
            """
        )

        upload_tab, camera_tab = st.tabs(
            ["Subir archivo", "Usar cámara"]
        )

        nonce = st.session_state.widget_nonce

        with upload_tab:
            uploaded_file = st.file_uploader(
                "Seleccione una imagen",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "bmp",
                    "tif",
                    "tiff",
                    "webp",
                ],
                help=(
                    "Formatos admitidos: JPG, PNG, BMP, TIFF y WEBP."
                ),
                key=f"uploaded_image_{nonce}",
            )

        with camera_tab:
            camera_file = st.camera_input(
                "Capture una fotografía",
                key=f"camera_image_{nonce}",
            )

        source_file = (
            camera_file
            if camera_file is not None
            else uploaded_file
        )

        current_image = None

        if source_file is not None:
            source_bytes = source_file.getvalue()
            current_hash = hashlib.sha256(
                source_bytes
            ).hexdigest()

            if (
                st.session_state.source_hash is not None
                and current_hash
                != st.session_state.source_hash
            ):
                st.session_state.result = None
                st.session_state.analyzed_image = None
                st.session_state.feedback_saved = False
                st.session_state.feedback_record = None
                st.session_state.feedback_message = ""
                st.session_state.show_correction = False
                st.session_state.corrected_labels = []

            st.session_state.source_hash = current_hash

            try:
                current_image = convert_to_pil(
                    io.BytesIO(source_bytes)
                )
                st.image(
                    current_image,
                    caption="Imagen preparada para el análisis",
                    use_container_width=True,
                )
            except Exception as error:
                st.error(
                    f"No fue posible abrir la imagen: {error}"
                )

        analyze_col, reset_col = st.columns(2)

        with analyze_col:
            analyze_clicked = st.button(
                "Analizar imagen",
                type="primary",
                use_container_width=True,
            )

        with reset_col:
            if st.button(
                "Nuevo análisis",
                use_container_width=True,
            ):
                current_nonce = st.session_state.widget_nonce
                reset_analysis()
                st.session_state.widget_nonce = current_nonce + 1
                st.session_state.control_code = ""
                st.session_state.control_moment = (
                    "Antes del procedimiento"
                )
                st.rerun()

        if analyze_clicked:
            if current_image is None:
                st.warning(
                    "Primero debe cargar o capturar una imagen."
                )
            else:
                with st.spinner(
                    "Analizando la imagen..."
                ):
                    try:
                        result = predict_instruments(
                            current_image,
                            model,
                        )
                        st.session_state.result = result
                        st.session_state.analyzed_image = (
                            current_image.copy()
                        )
                        st.session_state.feedback_saved = False
                        st.session_state.feedback_record = None
                        st.session_state.feedback_message = ""
                        st.session_state.show_correction = False
                        st.session_state.corrected_labels = (
                            result[
                                "instrumentos_detectados"
                            ].copy()
                        )
                    except Exception as error:
                        st.error(
                            "No fue posible completar el análisis. "
                            f"Detalle: {error}"
                        )

        st.html(
            """
            <div class="sv-note">
                <b>Captura recomendada:</b> separe los instrumentos,
                evite reflejos intensos, fondos metálicos y
                superposiciones que oculten su forma.
            </div>
            """
        )


with right:
    with st.container(border=True):
        st.html(
            """
            <div class="sv-section-kicker">Capacidades</div>
            <div class="sv-section-title">
                Instrumentos reconocibles
            </div>
            <div class="sv-section-copy">
                La versión actual está limitada a las siguientes
                cuatro clases.
            </div>
            """
        )
        st.html(
            class_cards_html()
        )

    with st.container(border=True):
        st.html(
            """
            <div class="sv-section-kicker">Resultado</div>
            <div class="sv-section-title">
                Revisión del análisis
            </div>
            <div class="sv-section-copy">
                La identificación es preliminar y debe validarse
                mediante inspección visual.
            </div>
            """
        )

        result = st.session_state.result

        st.html(
            result_status_html(result)
        )

        st.html(
            summary_html(
                result,
                code,
                moment,
                st.session_state.feedback_saved,
            )
        )

    with st.container(border=True):
        st.html(
            """
            <div class="sv-feedback-head">
                <div class="sv-feedback-title">
                    Validación para mejora del modelo
                </div>
                <div class="sv-feedback-badge">
                    HUMANO EN EL CICLO
                </div>
            </div>
            <div class="sv-feedback-copy">
                Confirme la predicción o seleccione las etiquetas
                correctas. La imagen y la revisión se guardarán
                para construir un conjunto validado.
            </div>
            """
        )

        observations = st.text_area(
            "Observaciones de la validación",
            placeholder=(
                "Ej.: reflejo intenso, instrumento parcialmente "
                "cubierto..."
            ),
            height=80,
            disabled=st.session_state.feedback_saved,
            key=f"observations_{st.session_state.widget_nonce}",
        )

        confirm_col, correct_col = st.columns(2)

        with confirm_col:
            confirm_clicked = st.button(
                "✓ Confirmar y guardar",
                type="primary",
                use_container_width=True,
                disabled=(
                    result is None
                    or st.session_state.feedback_saved
                ),
            )

        with correct_col:
            correct_clicked = st.button(
                "✎ Corregir resultado",
                use_container_width=True,
                disabled=(
                    result is None
                    or st.session_state.feedback_saved
                ),
            )

        if correct_clicked:
            st.session_state.show_correction = True

        if st.session_state.show_correction:
            corrected_labels = st.multiselect(
                "Instrumentos observados por el usuario",
                options=CLASS_NAMES,
                default=st.session_state.corrected_labels,
                help=(
                    "Seleccione todas las clases presentes. "
                    "Deje la selección vacía cuando no aparezca "
                    "ninguna de las cuatro clases."
                ),
                key=f"corrected_labels_widget_{st.session_state.widget_nonce}",
            )
            st.session_state.corrected_labels = (
                corrected_labels
            )

            save_correction_clicked = st.button(
                "Guardar corrección",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.feedback_saved,
            )
        else:
            save_correction_clicked = False

        if confirm_clicked and result is not None:
            try:
                record = save_validation(
                    result=result,
                    validated_labels=result[
                        "instrumentos_detectados"
                    ],
                    code=code,
                    moment=moment,
                    observations=observations,
                )
                st.session_state.feedback_saved = True
                st.session_state.feedback_record = record
                st.session_state.feedback_message = (
                    "Validación guardada correctamente."
                )
                st.session_state.show_correction = False
                st.rerun()
            except Exception as error:
                st.error(
                    "No fue posible guardar la validación. "
                    f"Detalle: {error}"
                )

        if save_correction_clicked and result is not None:
            try:
                record = save_validation(
                    result=result,
                    validated_labels=(
                        st.session_state.corrected_labels
                    ),
                    code=code,
                    moment=moment,
                    observations=observations,
                )
                st.session_state.feedback_saved = True
                st.session_state.feedback_record = record
                st.session_state.feedback_message = (
                    "Corrección guardada correctamente."
                )
                st.session_state.show_correction = False
                st.rerun()
            except Exception as error:
                st.error(
                    "No fue posible guardar la corrección. "
                    f"Detalle: {error}"
                )

        if st.session_state.feedback_saved:
            record = st.session_state.feedback_record or {}
            labels = record.get(
                "etiquetas_validadas",
                [],
            )
            labels_text = (
                ", ".join(labels)
                if labels
                else "Ninguno de los cuatro instrumentos"
            )

            st.success(
                f"{st.session_state.feedback_message} "
                f"Etiquetas validadas: {labels_text}."
            )
        elif result is None:
            st.info(
                "Analice una imagen antes de validar el resultado."
            )
        else:
            st.info(
                "La revisión aún no ha sido guardada."
            )


st.html(
    """
    <div class="sv-footer">
        <b>Aviso de uso:</b> SurgiVision AI es una prueba de
        concepto académica. No constituye un dispositivo médico,
        no reemplaza el conteo manual protocolizado ni la
        verificación establecida por el establecimiento de salud
        y no debe utilizarse para tomar decisiones clínicas.
    </div>
    """
)
