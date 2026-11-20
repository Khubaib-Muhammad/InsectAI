
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from PIL import Image

from utils.data_loader import load_dataset_metadata

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Dataset Explorer | InsectAI",
    page_icon="🪲",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = BASE_DIR / "artifacts" / "sample_images"

CLASS_NAMES = [
    "ant",
    "bee",
    "beetle",
    "butterfly",
    "grasshopper",
    "ladybug"
]

CLASS_LABELS = {
    "ant": "Ant",
    "bee": "Bee",
    "beetle": "Beetle",
    "butterfly": "Butterfly",
    "grasshopper": "Grasshopper",
    "ladybug": "Ladybug"
}

# --------------------------------
# Styling
# --------------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .class-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Header
# --------------------------------

st.title("🪲 Dataset Explorer")

st.markdown(
    "Explore the InsectAI image dataset, inspect class distributions, "
    "and browse representative insect images."
)

st.divider()

# --------------------------------
# Load Metadata
# --------------------------------

try:
    metadata = load_dataset_metadata()
except Exception as e:
    st.error(f"Unable to load dataset metadata: {e}")
    st.stop()

split_distribution = metadata["split_distribution"]
class_distribution = metadata["class_distribution"]

total_images = split_distribution["total"]
train_images = split_distribution["training"]
val_images = split_distribution["validation"]
test_images = split_distribution["testing"]
number_of_classes = metadata["number_of_classes"]

# --------------------------------
# Dataset Overview
# --------------------------------

st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Images", f"{total_images:,}")

with c2:
    st.metric("Insect Classes", number_of_classes)

with c3:
    st.metric("Training Images", f"{train_images:,}")

with c4:
    st.metric("Testing Images", f"{test_images:,}")

# --------------------------------
# Dataset Information
# --------------------------------

st.divider()

st.subheader("📁 Dataset Information")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown(f"**Dataset:** {metadata['dataset_name']}")
    st.markdown(f"**Source:** {metadata['source']}")
    st.markdown(f"**Number of Classes:** {number_of_classes}")

with info_col2:
    st.markdown("**Image Sizes:**")

    for model_name, image_size in metadata["image_size"].items():
        st.markdown(f"- {model_name}: {image_size}")

# --------------------------------
# Dataset Split Distribution
# --------------------------------

st.divider()

st.subheader("📦 Dataset Split Distribution")

split_df = pd.DataFrame({
    "Dataset Split": ["Training", "Validation", "Testing"],
    "Images": [train_images, val_images, test_images]
})

col1, col2 = st.columns(2)

with col1:
    fig = px.pie(
        split_df,
        names="Dataset Split",
        values="Images",
        hole=0.55,
        title="Dataset Split Percentage",
        template="plotly_dark"
    )

    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,} images<extra></extra>"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        split_df,
        x="Dataset Split",
        y="Images",
        color="Dataset Split",
        text="Images",
        title="Images per Dataset Split",
        template="plotly_dark"
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# Class Distribution
# --------------------------------

st.divider()

st.subheader("🐛 Class Distribution")

distribution_rows = []

for class_name in CLASS_NAMES:
    if class_name not in class_distribution:
        st.error(f"Class '{class_name}' is missing from metadata.")
        st.stop()

    class_data = class_distribution[class_name]

    distribution_rows.append({
        "Class": CLASS_LABELS[class_name],
        "Training": class_data["training"],
        "Validation": class_data["validation"],
        "Testing": class_data["testing"],
        "Total": (
            class_data["training"]
            + class_data["validation"]
            + class_data["testing"]
        )
    })

class_df = pd.DataFrame(distribution_rows)

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        class_df,
        x="Class",
        y=["Training", "Validation", "Testing"],
        barmode="group",
        title="Class Distribution Across Splits",
        template="plotly_dark"
    )

    fig.update_layout(
        xaxis_title="Insect Class",
        yaxis_title="Number of Images"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        class_df,
        x="Class",
        y="Total",
        color="Total",
        text="Total",
        title="Total Images per Class",
        template="plotly_dark"
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Insect Class",
        yaxis_title="Number of Images",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    class_df,
    use_container_width=True,
    hide_index=True
)

# --------------------------------
# Image Gallery
# --------------------------------

st.divider()

st.subheader("🖼️ Insect Image Gallery")

selected_class = st.selectbox(
    "Select Insect Class",
    CLASS_NAMES,
    format_func=lambda x: CLASS_LABELS[x]
)

class_folder = SAMPLE_DIR / selected_class

if not class_folder.exists():
    st.warning(f"Image folder not found: {class_folder}")
else:
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    image_files = sorted([
        path for path in class_folder.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ])

    if not image_files:
        st.warning("No sample images found in this folder.")
    else:
        st.markdown(
            f"**Showing {len(image_files)} sample images "
            f"for {CLASS_LABELS[selected_class]}**"
        )

        columns = st.columns(3)

        for index, image_path in enumerate(image_files):
            with columns[index % 3]:
                try:
                    image = Image.open(image_path).convert("RGB")

                    st.image(
                        image,
                        caption=image_path.name,
                        use_container_width=True
                    )

                    st.caption(
                        f"Dimensions: {image.width} × {image.height}"
                    )

                except Exception as e:
                    st.warning(
                        f"Could not load {image_path.name}: {e}"
                    )

# --------------------------------
# Dataset Split Details
# --------------------------------

st.divider()

st.subheader("📋 Dataset Split Details")

split_details = pd.DataFrame([
    {
        "Split": "Training",
        "Images": train_images,
        "Percentage": train_images / total_images * 100
    },
    {
        "Split": "Validation",
        "Images": val_images,
        "Percentage": val_images / total_images * 100
    },
    {
        "Split": "Testing",
        "Images": test_images,
        "Percentage": test_images / total_images * 100
    }
])

split_details["Percentage"] = (
    split_details["Percentage"].map(lambda x: f"{x:.2f}%")
)

st.dataframe(
    split_details,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "InsectAI | Dataset Exploration and Visualization"
)
