
import streamlit as st
import numpy as np
from PIL import Image

from utils.model_loader import load_model
from utils.preprocessing import preprocess_image
from utils.explainability import generate_gradcam

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Model Explainability | InsectAI",
    page_icon="🔍",
    layout="wide"
)

CLASS_NAMES = [
    "ant",
    "bee",
    "beetle",
    "butterfly",
    "grasshopper",
    "ladybug"
]

SUPPORTED_MODELS = [
    "CNN Version 1",
    "CNN Version 3"
]

st.title("🔍 Model Explainability")
st.caption(
    "Understand which image regions influence the CNN's prediction "
    "using Gradient-weighted Class Activation Mapping (Grad-CAM)."
)

# --------------------------------------------------
# MODEL AND IMAGE INPUT
# --------------------------------------------------

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    selected_model = st.selectbox(
        "Select CNN Model",
        SUPPORTED_MODELS
    )

with col2:
    uploaded_file = st.file_uploader(
        "Upload an Insect Image",
        type=["jpg", "jpeg", "png"]
    )

# --------------------------------------------------
# EXPLANATION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.divider()
    st.subheader("Uploaded Image")

    st.image(
        image,
        caption=uploaded_file.name,
        width=400
    )

    if st.button(
        "Generate Grad-CAM Explanation",
        type="primary",
        use_container_width=True
    ):

        try:
            with st.spinner(
                "Loading model and generating explanation..."
            ):

                model = load_model(selected_model)

                result = generate_gradcam(
                    image,
                    model,
                    selected_model
                )

            predicted_index = result["class_index"]

            predicted_class = CLASS_NAMES[
                predicted_index
            ]

            confidence = float(
                result["predictions"][predicted_index]
            )

            st.success("Grad-CAM generated successfully!")

            # ------------------------------------------
            # PREDICTION RESULT
            # ------------------------------------------

            st.divider()
            st.subheader("Classification Result")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Predicted Insect",
                predicted_class.capitalize()
            )

            col2.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            col3.metric(
                "Model Used",
                selected_model
            )

            # ------------------------------------------
            # VISUAL EXPLANATION
            # ------------------------------------------

            st.divider()
            st.subheader("Grad-CAM Visualization")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    image,
                    caption="Original Image",
                    use_container_width=True
                )

            with col2:
                st.image(
                    result["heatmap"],
                    caption="Grad-CAM Heatmap",
                    use_container_width=True
                )

            with col3:
                st.image(
                    result["overlay"],
                    caption="Heatmap Overlay",
                    use_container_width=True
                )

            # ------------------------------------------
            # EXPLANATION
            # ------------------------------------------

            st.divider()
            st.subheader("How to Interpret the Heatmap")

            st.markdown(
                """
                - **Warmer regions:** Areas receiving stronger
                  positive Grad-CAM activation for the predicted class.
                - **Cooler regions:** Areas receiving weaker activation.
                - **Overlay:** Combines the original image with
                  the generated heatmap.

                Grad-CAM provides an approximate visual explanation
                of the model's learned features. It does not prove
                that highlighted regions caused the prediction.
                """
            )

            # ------------------------------------------
            # CLASS PROBABILITIES
            # ------------------------------------------

            st.divider()
            st.subheader("Class Probability Distribution")

            probability_data = {
                "Insect": [
                    name.capitalize()
                    for name in CLASS_NAMES
                ],
                "Probability (%)": [
                    float(value) * 100
                    for value in result["predictions"]
                ]
            }

            import pandas as pd
            import plotly.express as px

            probability_df = pd.DataFrame(
                probability_data
            )

            fig = px.bar(
                probability_df,
                x="Insect",
                y="Probability (%)",
                color="Insect",
                text_auto=".2f",
                title="Predicted Probability for Each Class"
            )

            fig.update_layout(
                showlegend=False,
                yaxis_range=[0, 100]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ------------------------------------------
            # DOWNLOAD OVERLAY
            # ------------------------------------------

            st.divider()

            import io

            buffer = io.BytesIO()

            result["overlay"].save(
                buffer,
                format="PNG"
            )

            st.download_button(
                "⬇️ Download Grad-CAM Visualization",
                data=buffer.getvalue(),
                file_name="insectai_gradcam.png",
                mime="image/png",
                use_container_width=True
            )

        except Exception as e:
            st.error(
                f"Unable to generate Grad-CAM: {str(e)}"
            )

else:
    st.info(
        "Upload an insect image to generate a visual explanation."
    )

    st.markdown(
        """
        **Supported models:**
        - CNN Version 1
        - CNN Version 3

        MobileNetV2 will be supported separately.
        """
    )
