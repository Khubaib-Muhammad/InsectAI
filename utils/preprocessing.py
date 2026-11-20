
import numpy as np
from PIL import Image


MODEL_IMAGE_SIZES = {
    "CNN Version 1": (128, 128),
    "CNN Version 3": (128, 128),
    "MobileNetV2": (160, 160)
}


def preprocess_image(image, model_name):

    if model_name not in MODEL_IMAGE_SIZES:
        raise ValueError(f"Unknown model: {model_name}")

    image = image.convert("RGB")

    image = image.resize(
        MODEL_IMAGE_SIZES[model_name]
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array
