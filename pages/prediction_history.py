
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Prediction History | InsectAI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Prediction History & Analytics")
st.caption("Track, analyze, and review previous insect classification predictions.")

# --------------------------------------------------
# FILE CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_FILE = BASE_DIR / "artifacts" / "prediction_history.csv"

# --------------------------------------------------
# LOAD HISTORY
# --------------------------------------------------

if not HISTORY_FILE.exists():
    st.info("No prediction history found. Make a prediction first.")
    st.stop()

try:
    df = pd.read_csv(HISTORY_FILE)

    if df.empty:
        st.info("Prediction history is currently empty.")
        st.stop()

except Exception as e:
    st.error(f"Unable to read prediction history: {e}")
    st.stop()

required_columns = [
    "Timestamp",
    "Image",
    "Model",
    "Predicted Class",
    "Confidence (%)"
]

if not all(column in df.columns for column in required_columns):
    st.error("Prediction history has an unexpected format.")
    st.stop()

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

df["Confidence (%)"] = pd.to_numeric(
    df["Confidence (%)"],
    errors="coerce"
)

df = df.dropna(
    subset=["Timestamp", "Confidence (%)"]
)

if df.empty:
    st.info("No valid prediction records are available.")
    st.stop()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("History Filters")

models = sorted(df["Model"].dropna().unique().tolist())
classes = sorted(df["Predicted Class"].dropna().unique().tolist())

selected_models = st.sidebar.multiselect(
    "Filter by Model",
    models,
    default=models
)

selected_classes = st.sidebar.multiselect(
    "Filter by Insect Class",
    classes,
    default=classes
)

filtered_df = df[
    df["Model"].isin(selected_models)
    & df["Predicted Class"].isin(selected_classes)
].copy()

# --------------------------------------------------
# OVERVIEW METRICS
# --------------------------------------------------

st.divider()
st.subheader("Prediction Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Predictions",
    len(filtered_df)
)

col2.metric(
    "Models Used",
    filtered_df["Model"].nunique()
)

col3.metric(
    "Classes Detected",
    filtered_df["Predicted Class"].nunique()
)

average_confidence = (
    filtered_df["Confidence (%)"].mean()
    if not filtered_df.empty else 0
)

col4.metric(
    "Average Confidence",
    f"{average_confidence:.2f}%"
)

if filtered_df.empty:
    st.warning("No predictions match the selected filters.")
    st.stop()

# --------------------------------------------------
# PREDICTION DISTRIBUTION
# --------------------------------------------------

st.divider()
st.subheader("Prediction Distribution")

col1, col2 = st.columns(2)

class_counts = (
    filtered_df["Predicted Class"]
    .value_counts()
    .reset_index()
)

class_counts.columns = [
    "Insect Class",
    "Predictions"
]

with col1:
    fig_class = px.pie(
        class_counts,
        names="Insect Class",
        values="Predictions",
        hole=0.45,
        title="Predictions by Insect Class"
    )

    st.plotly_chart(
        fig_class,
        use_container_width=True
    )

with col2:
    model_counts = (
        filtered_df["Model"]
        .value_counts()
        .reset_index()
    )

    model_counts.columns = [
        "Model",
        "Predictions"
    ]

    fig_model = px.bar(
        model_counts,
        x="Model",
        y="Predictions",
        color="Model",
        text_auto=True,
        title="Predictions by Model"
    )

    fig_model.update_layout(showlegend=False)

    st.plotly_chart(
        fig_model,
        use_container_width=True
    )

# --------------------------------------------------
# CONFIDENCE DISTRIBUTION
# --------------------------------------------------

st.divider()
st.subheader("Prediction Confidence Analysis")

fig_confidence = px.histogram(
    filtered_df,
    x="Confidence (%)",
    nbins=15,
    color="Model",
    barmode="overlay",
    opacity=0.75,
    title="Distribution of Prediction Confidence"
)

st.plotly_chart(
    fig_confidence,
    use_container_width=True
)

# --------------------------------------------------
# PREDICTION TIMELINE
# --------------------------------------------------

st.divider()
st.subheader("Prediction Activity Over Time")

timeline = (
    filtered_df
    .assign(Date=filtered_df["Timestamp"].dt.date)
    .groupby("Date")
    .size()
    .reset_index(name="Predictions")
)

fig_timeline = px.line(
    timeline,
    x="Date",
    y="Predictions",
    markers=True,
    title="Daily Prediction Activity"
)

st.plotly_chart(
    fig_timeline,
    use_container_width=True
)

# --------------------------------------------------
# HISTORY TABLE
# --------------------------------------------------

st.divider()
st.subheader("Complete Prediction History")

display_df = filtered_df.sort_values(
    "Timestamp",
    ascending=False
).copy()

display_df["Timestamp"] = display_df[
    "Timestamp"
].dt.strftime("%Y-%m-%d %H:%M:%S")

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# DOWNLOAD HISTORY
# --------------------------------------------------

st.divider()

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "⬇️ Download Filtered History",
    data=csv_data,
    file_name="insectai_prediction_history.csv",
    mime="text/csv"
)

# --------------------------------------------------
# CLEAR HISTORY
# --------------------------------------------------

st.divider()
st.subheader("Manage Prediction History")

with st.expander("Clear Prediction History"):
    st.warning(
        "This permanently deletes all locally saved prediction records."
    )

    confirm = st.checkbox(
        "I understand that this action cannot be undone."
    )

    if st.button(
        "Clear All History",
        type="primary",
        disabled=not confirm
    ):
        HISTORY_FILE.unlink(missing_ok=True)
        st.success("Prediction history cleared.")
        st.rerun()

st.caption(
    "InsectAI | Prediction records are stored locally in artifacts/prediction_history.csv."
)
