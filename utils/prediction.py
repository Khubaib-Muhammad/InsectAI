
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from utils.model_loader import load_model
from utils.preprocessing import preprocess_image

CLASS_NAMES = [
    "ant",
    "bee",
    "beetle",
    "butterfly",
    "grasshopper",
    "ladybug"
]

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_FILE = BASE_DIR / "artifacts" / "prediction_history.csv"


def save_prediction(filename, model_name, predicted_class, confidence):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    new_record = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Image": filename,
        "Model": model_name,
        "Predicted Class": predicted_class,
        "Confidence (%)": round(confidence * 100, 2)
    }])

    if HISTORY_FILE.exists():
        new_record.to_csv(
            HISTORY_FILE,
            mode="a",
            header=False,
            index=False
        )
    else:
        new_record.to_csv(
            HISTORY_FILE,
            index=False
        )


def predict_insect(image, model_name, filename="uploaded_image"):
    model = load_model(model_name)

    processed_image = preprocess_image(image, model_name)

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    probabilities = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    save_prediction(
        filename,
        model_name,
        predicted_class,
        confidence
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }
