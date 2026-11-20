
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.data_loader import load_training_history

st.set_page_config(
    page_title="Training Analysis | InsectAI",
    page_icon="📈",
    layout="wide"
)

# -------------------------------
# Custom Styling
# -------------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .section-heading {
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Page Header
# -------------------------------

st.title("📈 Model Training Analysis")

st.markdown(
    "Explore training progress, validation performance, "
    "convergence behavior, and overfitting across InsectAI models."
)

st.divider()

# -------------------------------
# Load Training History
# -------------------------------

try:
    training_history = load_training_history()
except Exception as e:
    st.error(f"Unable to load training history: {e}")
    st.stop()

model_names = list(training_history.keys())

if not model_names:
    st.error("No model training histories found.")
    st.stop()

# -------------------------------
# Model Selection
# -------------------------------

selected_model = st.selectbox(
    "Select Model",
    model_names
)

model_data = training_history[selected_model]
metrics = model_data["metrics"]

required_metrics = [
    "accuracy",
    "val_accuracy",
    "loss",
    "val_loss"
]

missing = [key for key in required_metrics if key not in metrics]

if missing:
    st.error(f"Missing training metrics: {missing}")
    st.stop()

# Validate history lengths
lengths = [len(metrics[key]) for key in required_metrics]

if len(set(lengths)) != 1:
    st.error("Training metric arrays have inconsistent lengths.")
    st.stop()

total_epochs = model_data["total_epochs"]

if total_epochs != lengths[0]:
    st.warning(
        f"Recorded epochs ({total_epochs}) differ from "
        f"metric history length ({lengths[0]})."
    )

epochs = list(range(1, lengths[0] + 1))

# -------------------------------
# Best Training Results
# -------------------------------

best_val_accuracy = model_data["best_validation_accuracy"] * 100
best_accuracy_epoch = model_data["best_accuracy_epoch"]

best_val_loss = model_data["best_validation_loss"]
best_loss_epoch = model_data["best_loss_epoch"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Epochs",
        total_epochs
    )

with col2:
    st.metric(
        "Best Validation Accuracy",
        f"{best_val_accuracy:.2f}%"
    )

with col3:
    st.metric(
        "Best Accuracy Epoch",
        best_accuracy_epoch
    )

with col4:
    st.metric(
        "Best Validation Loss",
        f"{best_val_loss:.4f}"
    )

st.divider()

# -------------------------------
# Accuracy Chart
# -------------------------------

st.subheader("🎯 Training and Validation Accuracy")

accuracy_fig = go.Figure()

accuracy_fig.add_trace(go.Scatter(
    x=epochs,
    y=[v * 100 for v in metrics["accuracy"]],
    mode="lines+markers",
    name="Training Accuracy",
    line=dict(width=2)
))

accuracy_fig.add_trace(go.Scatter(
    x=epochs,
    y=[v * 100 for v in metrics["val_accuracy"]],
    mode="lines+markers",
    name="Validation Accuracy",
    line=dict(width=2)
))

accuracy_fig.add_vline(
    x=best_accuracy_epoch,
    line_dash="dash",
    annotation_text="Best Validation Accuracy",
    annotation_position="top"
)

accuracy_fig.update_layout(
    title=f"{selected_model} - Accuracy over Epochs",
    xaxis_title="Epoch",
    yaxis_title="Accuracy (%)",
    template="plotly_dark",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.12)
)

st.plotly_chart(
    accuracy_fig,
    use_container_width=True
)

# -------------------------------
# Loss Chart
# -------------------------------

st.subheader("📉 Training and Validation Loss")

loss_fig = go.Figure()

loss_fig.add_trace(go.Scatter(
    x=epochs,
    y=metrics["loss"],
    mode="lines+markers",
    name="Training Loss",
    line=dict(width=2)
))

loss_fig.add_trace(go.Scatter(
    x=epochs,
    y=metrics["val_loss"],
    mode="lines+markers",
    name="Validation Loss",
    line=dict(width=2)
))

loss_fig.add_vline(
    x=best_loss_epoch,
    line_dash="dash",
    annotation_text="Best Validation Loss",
    annotation_position="top"
)

loss_fig.update_layout(
    title=f"{selected_model} - Loss over Epochs",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    template="plotly_dark",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.12)
)

st.plotly_chart(
    loss_fig,
    use_container_width=True
)

st.divider()

# -------------------------------
# Training Analysis
# -------------------------------

st.subheader("🔍 Training Performance Insights")

final_train_accuracy = metrics["accuracy"][-1] * 100
final_val_accuracy = metrics["val_accuracy"][-1] * 100

accuracy_gap = final_train_accuracy - final_val_accuracy

final_train_loss = metrics["loss"][-1]
final_val_loss = metrics["val_loss"][-1]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Final Training Accuracy",
        f"{final_train_accuracy:.2f}%"
    )

with col2:
    st.metric(
        "Final Validation Accuracy",
        f"{final_val_accuracy:.2f}%"
    )

with col3:
    st.metric(
        "Accuracy Gap",
        f"{accuracy_gap:.2f} percentage points"
    )

st.markdown("#### Observations")

if accuracy_gap > 15:
    st.warning(
        "The training accuracy is considerably higher than "
        "validation accuracy. This may indicate overfitting."
    )
elif accuracy_gap > 8:
    st.info(
        "There is a noticeable training-validation accuracy gap. "
        "Monitor generalization performance."
    )
else:
    st.success(
        "Training and validation accuracy are relatively close."
    )

if final_val_loss > best_val_loss * 1.5:
    st.warning(
        "Validation loss has increased considerably since its "
        "best epoch. The model may have started overfitting."
    )

# -------------------------------
# Epoch-by-Epoch Data
# -------------------------------

st.divider()

st.subheader("📋 Epoch Performance Details")

epoch_df = pd.DataFrame({
    "Epoch": epochs,
    "Training Accuracy (%)": [
        v * 100 for v in metrics["accuracy"]
    ],
    "Validation Accuracy (%)": [
        v * 100 for v in metrics["val_accuracy"]
    ],
    "Training Loss": metrics["loss"],
    "Validation Loss": metrics["val_loss"]
})

st.dataframe(
    epoch_df.round(4),
    use_container_width=True,
    hide_index=True
)

csv_data = epoch_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Epoch Data",
    data=csv_data,
    file_name=f"{selected_model.replace(' ', '_')}_training_history.csv",
    mime="text/csv"
)

# -------------------------------
# Training Stages
# -------------------------------

st.divider()

st.subheader("⚙️ Training Stages")

stages = model_data.get("training_stages", [])

if stages:
    stage_df = pd.DataFrame(stages)
    st.dataframe(
        stage_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No training-stage information available.")

st.caption(
    "InsectAI | Training History and Model Optimization Analysis"
)
