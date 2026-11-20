
from pathlib import Path
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATHS = {
    "CNN Version 1": MODEL_DIR / "cnn_version_1.keras",
    "CNN Version 3": MODEL_DIR / "cnn_version_3.keras",
    "MobileNetV2": MODEL_DIR / "mobilenetv2_insect_classifier.keras"
}


@st.cache_resource
def load_model(model_name):

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = MODEL_PATHS[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model


def get_model_names():
    return list(MODEL_PATHS.keys())