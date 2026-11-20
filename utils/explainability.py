
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import colormaps

from utils.preprocessing import preprocess_image


TARGET_LAYERS = {
    "CNN Version 1": "conv2d_2",
    "CNN Version 3": "activation_3"
}


def build_grad_model(model, model_name):
    """
    Build a functional model that returns:
    1. Target convolutional layer output
    2. Final prediction output
    """

    input_sizes = {
        "CNN Version 1": (128, 128, 3),
        "CNN Version 3": (128, 128, 3)
    }

    inputs = tf.keras.Input(shape=input_sizes[model_name])

    x = inputs
    target_output = None

    inference_layers = (
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.Dropout,
        tf.keras.layers.RandomFlip,
        tf.keras.layers.RandomRotation,
        tf.keras.layers.RandomZoom
    )

    for layer in model.layers:

        if isinstance(layer, inference_layers):
            x = layer(x, training=False)
        else:
            x = layer(x)

        if layer.name == TARGET_LAYERS[model_name]:
            target_output = x

    if target_output is None:
        raise ValueError(
            f"Target layer '{TARGET_LAYERS[model_name]}' not found."
        )

    return tf.keras.Model(
        inputs=inputs,
        outputs=[target_output, x]
    )


def generate_gradcam(image, model, model_name, class_index=None):

    if model_name not in TARGET_LAYERS:
        raise ValueError(
            "Grad-CAM is not configured for this model."
        )

    # Preprocess uploaded image
    processed_image = preprocess_image(image, model_name)

    input_tensor = tf.convert_to_tensor(
        processed_image,
        dtype=tf.float32
    )

    # Build functional Grad-CAM model
    grad_model = build_grad_model(model, model_name)

    # Calculate gradients
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            input_tensor,
            training=False
        )

        if class_index is None:
            class_index = int(
                tf.argmax(predictions[0]).numpy()
            )

        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise ValueError(
            "Unable to calculate gradients for the selected layer."
        )

    # Global average pooling of gradients
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    # Generate heatmap
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = tf.math.divide_no_nan(
        heatmap,
        tf.reduce_max(heatmap)
    )

    heatmap = heatmap.numpy()

    # Resize heatmap to original image dimensions
    original_image = image.convert("RGB")

    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    # Apply color map
    heatmap_array = (
        np.asarray(heatmap_image, dtype=np.float32) / 255.0
    )

    colored_heatmap = colormaps["jet"](
        heatmap_array
    )[:, :, :3]

    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )

    colored_heatmap_image = Image.fromarray(
        colored_heatmap
    )

    # Overlay heatmap on original image
    overlay = Image.blend(
        original_image,
        colored_heatmap_image,
        alpha=0.4
    )

    return {
        "heatmap": colored_heatmap_image,
        "overlay": overlay,
        "predictions": predictions.numpy()[0],
        "class_index": int(class_index)
    }
