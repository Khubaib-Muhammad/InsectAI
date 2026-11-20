
import json
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


@st.cache_data
def load_json(filename):

    file_path = ARTIFACT_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_dataset_metadata():
    return load_json("dataset_metadata.json")


def load_model_metrics():
    return load_json("model_metrics.json")


def load_training_history():
    return load_json("training_history.json")


def load_model_metadata():
    return load_json("model_metadata.json")
