
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import (
    load_dataset_metadata,
    load_model_metrics,
    load_training_history
)


st.set_page_config(
    page_title="InsectAI | Dashboard",
    page_icon="🐞",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------------
# CUSTOM STYLING
# -------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #0E1525;
    }

    [data-testid="stMetric"] {
        background-color: #192438;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #263750;
    }

    h1, h2, h3 {
        color: #F1F5F9;
    }

    p, label {
        color: #CBD5E1;
    }

    [data-testid="stSidebar"] {
        background-color: #111C30;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------
# LOAD ARTIFACTS
# -------------------------------

try:
    dataset = load_dataset_metadata()
    metrics = load_model_metrics()
    history = load_training_history()

except Exception as error:
    st.error(f"Unable to load dashboard artifacts: {error}")
    st.stop()


# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.title("🐞 InsectAI")

st.sidebar.caption(
    "Intelligent Insect Classification Platform"
)

st.sidebar.divider()

st.sidebar.markdown("### Navigation")

st.sidebar.info(
    "Use the pages in the sidebar to explore the application."
)


# -------------------------------
# HEADER
# -------------------------------

st.title("InsectAI Dashboard")

st.markdown(
    "An intelligent deep learning platform for "
    "six-class insect image classification."
)

st.divider()


# -------------------------------
# DATASET METRICS
# -------------------------------

split = dataset["split_distribution"]

total_images = split["total"]
total_classes = dataset["number_of_classes"]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Images", f"{total_images:,}")
col2.metric("Insect Classes", total_classes)
col3.metric("Training Images", f"{split['training']:,}")
col4.metric("Test Images", f"{split['testing']:,}")


# -------------------------------
# DATASET DISTRIBUTION
# -------------------------------

st.subheader("Dataset Overview")

left, right = st.columns(2)

class_distribution = dataset["class_distribution"]

distribution_df = pd.DataFrame([
    {
        "Class": class_name.capitalize(),
        "Training": values["training"],
        "Validation": values["validation"],
        "Testing": values["testing"]
    }
    for class_name, values in class_distribution.items()
])

with left:

    st.markdown("#### Class Distribution")

    fig = px.bar(
        distribution_df,
        x="Class",
        y=["Training", "Validation", "Testing"],
        barmode="group",
        title="Images per Class",
        template="plotly_dark"
    )

    fig.update_layout(
        height=400,
        legend_title="Dataset Split"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    st.markdown("#### Dataset Split")

    split_df = pd.DataFrame({
        "Split": ["Training", "Validation", "Testing"],
        "Images": [
            split["training"],
            split["validation"],
            split["testing"]
        ]
    })

    fig = px.pie(
        split_df,
        names="Split",
        values="Images",
        hole=0.55,
        template="plotly_dark"
    )

    fig.update_layout(height=400)

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# -------------------------------
# MODEL PERFORMANCE
# -------------------------------

st.divider()

st.subheader("Model Performance Comparison")

# Expected metric format from model_metrics.json:
# {
#   "CNN Version 1": {
#       "accuracy": 0.0,
#       "precision": 0.0,
#       "recall": 0.0,
#       "f1_score": 0.0
#   }
# }

performance_rows = []

for model_name, values in metrics.items():

    performance_rows.append({
        "Model": model_name,
        "Accuracy": values["accuracy"] * 100,
        "Precision": values["precision"] * 100,
        "Recall": values["recall"] * 100,
        "F1 Score": values["f1_score"] * 100
    })

performance_df = pd.DataFrame(performance_rows)

st.dataframe(
    performance_df.round(2),
    use_container_width=True,
    hide_index=True
)

fig = px.bar(
    performance_df,
    x="Model",
    y=["Accuracy", "Precision", "Recall", "F1 Score"],
    barmode="group",
    template="plotly_dark",
    title="Test Performance by Model"
)

fig.update_layout(height=450)

st.plotly_chart(
    fig,
    use_container_width=True
)


# -------------------------------
# TRAINING HIGHLIGHTS
# -------------------------------

st.divider()

st.subheader("Training Highlights")

history_rows = []

for model_name, values in history.items():

    history_rows.append({
        "Model": model_name,
        "Best Validation Accuracy":
            values["best_validation_accuracy"] * 100,
        "Best Validation Loss":
            values["best_validation_loss"],
        "Total Epochs":
            values["total_epochs"]
    })

history_df = pd.DataFrame(history_rows)

st.dataframe(
    history_df.round(3),
    use_container_width=True,
    hide_index=True
)


# -------------------------------
# FOOTER
# -------------------------------

st.divider()

st.caption(
    "InsectAI | Deep Learning Image Classification | "
    "Six Insect Categories"
)
