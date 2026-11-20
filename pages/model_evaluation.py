
import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

from utils.data_loader import load_json

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Model Evaluation | InsectAI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Evaluation")
st.caption(
    "Detailed evaluation of insect classification models "
    "using confusion matrices and class-wise performance metrics."
)

# --------------------------------------------------
# LOAD ARTIFACTS
# --------------------------------------------------

try:
    classification_reports = load_json("classification_reports.json")
    confusion_matrices = load_json("confusion_matrices.json")

except Exception as e:
    st.error(f"Unable to load evaluation artifacts: {e}")
    st.stop()

MODEL_NAMES = [
    "CNN Version 1",
    "CNN Version 3",
    "MobileNetV2"
]

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

selected_model = st.selectbox(
    "Select Model",
    MODEL_NAMES
)

if selected_model not in classification_reports:
    st.error("Classification report not available for this model.")
    st.stop()

if selected_model not in confusion_matrices:
    st.error("Confusion matrix not available for this model.")
    st.stop()

report = classification_reports[selected_model]
matrix_data = confusion_matrices[selected_model]

class_names = matrix_data["class_names"]
matrix = matrix_data["matrix"]

# --------------------------------------------------
# OVERALL PERFORMANCE
# --------------------------------------------------

st.divider()
st.subheader("Overall Classification Performance")

accuracy = report.get("accuracy", 0)
macro_avg = report.get("macro avg", {})
weighted_avg = report.get("weighted avg", {})

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy * 100:.2f}%"
)

col2.metric(
    "Macro Precision",
    f"{macro_avg.get('precision', 0) * 100:.2f}%"
)

col3.metric(
    "Macro Recall",
    f"{macro_avg.get('recall', 0) * 100:.2f}%"
)

col4.metric(
    "Macro F1 Score",
    f"{macro_avg.get('f1-score', 0) * 100:.2f}%"
)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.divider()
st.subheader("Confusion Matrix")

st.write(
    "The confusion matrix shows the number of correct and "
    "incorrect predictions for each insect class."
)

fig_cm = px.imshow(
    matrix,
    x=class_names,
    y=class_names,
    text_auto=True,
    color_continuous_scale="Blues",
    labels={
        "x": "Predicted Class",
        "y": "Actual Class",
        "color": "Number of Images"
    },
    aspect="auto"
)

fig_cm.update_layout(
    height=550,
    xaxis_title="Predicted Class",
    yaxis_title="Actual Class"
)

st.plotly_chart(fig_cm, use_container_width=True)

st.caption(
    "Diagonal values represent correct predictions. "
    "Off-diagonal values represent misclassified images."
)

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

st.divider()
st.subheader("Class-wise Classification Report")

class_report_data = []

for class_name in class_names:
    metrics = report.get(class_name, {})

    class_report_data.append({
        "Insect Class": class_name.capitalize(),
        "Precision (%)": metrics.get("precision", 0) * 100,
        "Recall (%)": metrics.get("recall", 0) * 100,
        "F1 Score (%)": metrics.get("f1-score", 0) * 100,
        "Support": int(metrics.get("support", 0))
    })

class_df = pd.DataFrame(class_report_data)

st.dataframe(
    class_df.style.format({
        "Precision (%)": "{:.2f}",
        "Recall (%)": "{:.2f}",
        "F1 Score (%)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# CLASS-WISE METRIC COMPARISON
# --------------------------------------------------

st.divider()
st.subheader("Class-wise Metric Comparison")

selected_metric = st.selectbox(
    "Select Evaluation Metric",
    ["Precision (%)", "Recall (%)", "F1 Score (%)"]
)

fig_metrics = px.bar(
    class_df,
    x="Insect Class",
    y=selected_metric,
    color="Insect Class",
    text_auto=".2f",
    title=f"{selected_metric} by Insect Class"
)

fig_metrics.update_layout(
    showlegend=False,
    yaxis_title=selected_metric,
    xaxis_title="Insect Class",
    yaxis_range=[0, 100]
)

st.plotly_chart(fig_metrics, use_container_width=True)

# --------------------------------------------------
# CLASSIFICATION INSIGHTS
# --------------------------------------------------

st.divider()
st.subheader("Classification Insights")

best_class = class_df.loc[
    class_df["F1 Score (%)"].idxmax()
]

worst_class = class_df.loc[
    class_df["F1 Score (%)"].idxmin()
]

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"**Best Performing Class:** {best_class['Insect Class']}\n\n"
        f"F1 Score: {best_class['F1 Score (%)']:.2f}%"
    )

with col2:
    st.warning(
        f"**Weakest Performing Class:** {worst_class['Insect Class']}\n\n"
        f"F1 Score: {worst_class['F1 Score (%)']:.2f}%"
    )

# --------------------------------------------------
# MACRO AND WEIGHTED AVERAGES
# --------------------------------------------------

st.divider()
st.subheader("Average Performance")

average_df = pd.DataFrame([
    {
        "Average Type": "Macro Average",
        "Precision (%)": macro_avg.get("precision", 0) * 100,
        "Recall (%)": macro_avg.get("recall", 0) * 100,
        "F1 Score (%)": macro_avg.get("f1-score", 0) * 100
    },
    {
        "Average Type": "Weighted Average",
        "Precision (%)": weighted_avg.get("precision", 0) * 100,
        "Recall (%)": weighted_avg.get("recall", 0) * 100,
        "F1 Score (%)": weighted_avg.get("f1-score", 0) * 100
    }
])

st.dataframe(
    average_df.style.format({
        "Precision (%)": "{:.2f}",
        "Recall (%)": "{:.2f}",
        "F1 Score (%)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# DOWNLOAD REPORT
# --------------------------------------------------

st.divider()

csv_data = class_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Classification Report",
    data=csv_data,
    file_name=f"{selected_model.lower().replace(' ', '_')}_evaluation.csv",
    mime="text/csv"
)

st.success("Model evaluation loaded successfully.")
