
import streamlit as st
import pandas as pd
import plotly.express as px

from PIL import Image

from utils.prediction import predict_insect
from utils.model_loader import get_model_names


st.set_page_config(
    page_title="Live Classification | InsectAI",
    page_icon="🐞",
    layout="wide"
)

st.title("🔬 Live Insect Classification")

st.markdown(
    "Upload an insect image and let our trained deep learning "
    "models identify its class."
)

st.divider()

# Sidebar settings
st.sidebar.header("Prediction Settings")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    get_model_names()
)

show_probabilities = st.sidebar.checkbox(
    "Show class probability distribution",
    value=True
)

# Image upload
uploaded_file = st.file_uploader(
    "Upload an insect image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(
            image,
            use_container_width=True
        )

    with col2:
        st.subheader("Classification Results")

        if st.button(
            "Classify Insect",
            type="primary",
            use_container_width=True
        ):

            try:
                with st.spinner("Analyzing image..."):

                    result = predict_insect(
                                image,
                                selected_model,
                                uploaded_file.name
                            )

                predicted_class = result["predicted_class"]
                confidence = result["confidence"]

                st.success("Classification completed!")

                st.metric(
                    "Predicted Insect",
                    predicted_class.capitalize()
                )

                st.metric(
                    "Model Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.caption(
                    "Confidence represents the model's predicted "
                    "score, not a guarantee of correctness."
                )

                if show_probabilities:

                    st.divider()
                    st.subheader("Class Score Distribution")

                    probability_df = pd.DataFrame({
                        "Insect": list(result["probabilities"].keys()),
                        "Score": [
                            value * 100
                            for value in result["probabilities"].values()
                        ]
                    }).sort_values(
                        "Score",
                        ascending=True
                    )

                    fig = px.bar(
                        probability_df,
                        x="Score",
                        y="Insect",
                        orientation="h",
                        text_auto=".2f",
                        labels={
                            "Score": "Model Score (%)",
                            "Insect": "Insect Class"
                        },
                        title="Prediction Scores"
                    )

                    fig.update_layout(
                        height=400,
                        xaxis_range=[0, 100],
                        showlegend=False
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            except Exception as error:
                st.error(
                    f"Prediction failed: {error}"
                )

else:
    st.info(
        "Upload an image to get started."
    )

    st.markdown(
        """
        **Supported formats:** JPG, JPEG, PNG, WEBP

        **Available models:**
        - CNN Version 1
        - CNN Version 3
        - MobileNetV2
        """
    )
