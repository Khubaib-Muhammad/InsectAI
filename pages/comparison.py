
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import (
    load_model_metrics,
    load_model_metadata
)

st.set_page_config(
    page_title="Model Comparison | InsectAI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Comparison")
st.write(
    "Compare the performance and architecture of all trained models."
)

st.divider()

try:
    metrics = load_model_metrics()
    metadata = load_model_metadata()

except Exception as error:
    st.error(f"Unable to load model information: {error}")
    st.stop()



# PERFORMANCE COMPARISON

st.subheader("Performance Metrics")

rows = []

for model_name, values in metrics.items():

    rows.append({
        "Model": model_name,
        "Accuracy (%)": values["accuracy"] * 100,
        "Precision (%)": values["precision"] * 100,
        "Recall (%)": values["recall"] * 100,
        "F1 Score (%)": values["f1_score"] * 100,
        "Test Loss": values["loss"],
        "Parameters": values["parameters"],
        "Inference (ms/image)": values[
            "average_inference_time_ms_per_image"
        ]
    })

df = pd.DataFrame(rows)

st.dataframe(
    df.round(3),
    use_container_width=True,
    hide_index=True
)

st.subheader("Accuracy Comparison")

fig = px.bar(
    df,
    x="Model",
    y="Accuracy (%)",
    color="Model",
    text_auto=".2f",
    template="plotly_dark",
    title="Test Accuracy"
)

fig.update_layout(height=420)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("F1 Score Comparison")

fig = px.bar(
    df,
    x="Model",
    y="F1 Score (%)",
    color="Model",
    text_auto=".2f",
    template="plotly_dark",
    title="Test F1 Score"
)

fig.update_layout(height=420)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------
# METRIC SELECTOR
# ---------------------------------
# Metric comparison
st.subheader("Metric Comparison")

metric_options = {
    "Accuracy": "Accuracy (%)",
    "Precision": "Precision (%)",
    "Recall": "Recall (%)",
    "F1 Score": "F1 Score (%)"
}

selected_metric = st.selectbox(
    "Select Metric",
    list(metric_options.keys())
)

selected_column = metric_options[selected_metric]

fig = px.bar(
    df,
    x="Model",
    y=selected_column,
    color="Model",
    text=selected_column,
    title=f"{selected_metric} Comparison",
    template="plotly_dark"
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig.update_layout(
    yaxis_title=selected_column,
    xaxis_title="Model",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)
# ---------------------------------
# MODEL ARCHITECTURE COMPARISON
# ---------------------------------

st.divider()

st.subheader("Architecture Comparison")

architecture_rows = []

for model_name, values in metadata.items():

    architecture_rows.append({
        "Model": model_name,
        "Parameters": values["total_parameters"],
        "Layers": values["number_of_layers"],
        "Input Shape": values["input_shape"],
        "Trainable Parameters": values["trainable_parameters"]
    })

architecture_df = pd.DataFrame(architecture_rows)

st.dataframe(
    architecture_df,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------
# PARAMETER COMPARISON
# ---------------------------------

fig = px.bar(
    architecture_df,
    x="Model",
    y="Parameters",
    color="Model",
    text="Parameters",
    template="plotly_dark",
    title="Model Parameter Count"
)

fig.update_layout(
    height=400,
    yaxis_title="Number of Parameters"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------------
# BEST MODEL
# ---------------------------------
st.subheader("🏆 Best Model Summary")

best_accuracy = df.loc[df["Accuracy (%)"].idxmax()]
best_f1 = df.loc[df["F1 Score (%)"].idxmax()]
fastest_model = df.loc[df["Inference (ms/image)"].idxmin()]
smallest_model = df.loc[df["Parameters"].idxmin()]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Best Accuracy",
        best_accuracy["Model"],
        f'{best_accuracy["Accuracy (%)"]:.2f}%'
    )

with col2:
    st.metric(
        "Best F1 Score",
        best_f1["Model"],
        f'{best_f1["F1 Score (%)"]:.2f}%'
    )

with col3:
    st.metric(
        "Fastest Model",
        fastest_model["Model"],
        f'{fastest_model["Inference (ms/image)"]:.3f} ms'
    )

with col4:
    st.metric(
        "Smallest Model",
        smallest_model["Model"],
        f'{smallest_model["Parameters"]:,} parameters'
    )