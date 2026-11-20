# 🦋 InsectAI — Intelligent Insect Classification Platform

**InsectAI** is a deep learning-based image classification platform designed to identify insects from uploaded images. It uses custom Convolutional Neural Networks (CNNs) and transfer learning with MobileNetV2 to classify images into six insect categories.

Built with **TensorFlow, Keras, Python, and Streamlit**, the platform provides an interactive environment for insect classification, model performance analysis, training visualization, prediction tracking, and explainable AI using Grad-CAM.

🔗 **Live Application:** [InsectAI – Streamlit App](https://insectai-qnsyy75tjbe2pkvbv9mhfe.streamlit.app/model_explainability)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Insect Categories](#-insect-categories)
- [Dataset](#-dataset)
- [Deep Learning Models](#-deep-learning-models)
- [Model Performance](#-model-performance)
- [Application Pages](#-application-pages)
- [Grad-CAM Explainability](#-grad-cam-explainability)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Running the Application](#-running-the-application)
- [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

InsectAI explores the application of deep learning and computer vision to automated insect image classification.

The project implements and compares three deep learning architectures:

1. **CNN Version 1** – A custom convolutional neural network.
2. **CNN Version 3** – An improved custom CNN architecture.
3. **MobileNetV2** – A pretrained convolutional neural network using transfer learning.

The trained models are integrated into a multi-page Streamlit application where users can upload insect images, generate predictions, explore model performance, inspect training history, and understand model decisions through Grad-CAM visualizations.

The platform supports six insect classes and provides a complete workflow from image classification to model explainability.

## ✨ Key Features

- **Multi-Class Insect Classification:** Classifies uploaded images into six insect categories.
- **Multiple Deep Learning Models:** Supports three trained architectures.
- **Model Comparison:** Compares model accuracy, precision, recall, F1-score, loss, parameter count, and inference time.
- **Training Analysis:** Visualizes training and validation accuracy and loss across epochs.
- **Dataset Exploration:** Displays dataset information, class distribution, and sample images.
- **Model Evaluation:** Provides confusion matrices, classification reports, and class-wise performance analysis.
- **Prediction History:** Records previous predictions with timestamps, model names, predicted classes, and confidence scores.
- **Grad-CAM Explainability:** Generates heatmaps and overlays to visualize image regions influencing CNN predictions.
- **Interactive Dashboard:** Provides an accessible interface through Streamlit.
- **Prediction Analytics:** Allows users to explore prediction history and confidence distributions.

## 🐛 Insect Categories

The platform supports six insect categories:

| Class | Insect |
|---|---|
| 1 | Ant |
| 2 | Bee |
| 3 | Beetle |
| 4 | Butterfly |
| 5 | Grasshopper |
| 6 | Ladybug |

## 📂 Dataset

The project uses the [Insects Dataset](https://www.kaggle.com/datasets/baxtiyorbotiraliyev/insects) from Kaggle.

A balanced subset of **7,200 images** was selected for the classification task.

### Dataset Distribution

| Dataset Split | Images | Percentage |
|---|---:|---:|
| Training | 5,040 | 70% |
| Validation | 1,080 | 15% |
| Testing | 1,080 | 15% |
| **Total** | **7,200** | **100%** |

Each insect category contains:

- Training: 840 images
- Validation: 180 images
- Testing: 180 images

### Image Preprocessing

The preprocessing pipeline includes:

- Image loading and RGB conversion.
- Image resizing according to model input requirements.
- Pixel normalization to the range [0, 1].
- Data augmentation during training.
- Stratified dataset splitting to maintain class distribution.

### Data Augmentation

The training pipeline uses:

- Random horizontal flipping.
- Random rotation.
- Random zooming.

These transformations help improve model generalization and reduce overfitting.

## 🧠 Deep Learning Models

Three architectures were implemented and evaluated.

### 1. CNN Version 1

A custom CNN architecture developed for six-class insect classification.

**Architecture components:**

- Convolutional layers.
- ReLU activation.
- Max pooling.
- Flatten layer.
- Fully connected dense layers.
- Dropout regularization.
- Softmax output layer.

**Configuration:**

- Input size: 128 × 128 × 3
- Output classes: 6
- Parameters: 4,288,454

### 2. CNN Version 3

An improved custom CNN architecture designed with additional feature extraction and regularization components.

**Architecture components:**

- Data augmentation layers.
- Convolutional layers.
- Batch normalization.
- ReLU activation.
- Max pooling.
- Global average pooling.
- Dense layers.
- Dropout regularization.
- Softmax output layer.

**Configuration:**

- Input size: 128 × 128 × 3
- Output classes: 6
- Parameters: 424,006

### 3. MobileNetV2

MobileNetV2 is a pretrained convolutional neural network originally trained on ImageNet.

Transfer learning was applied by:

- Loading pretrained ImageNet weights.
- Initially freezing the pretrained feature extraction layers.
- Adding custom classification layers.
- Applying global average pooling.
- Using dropout and L2 regularization.
- Fine-tuning selected deeper layers.
- Keeping batch normalization layers frozen during fine-tuning.

**Configuration:**

- Input size: 160 × 160 × 3
- Output classes: 6
- Parameters: 2,422,726

## 📊 Model Performance

All three models were evaluated on the same held-out test dataset containing **1,080 images**, with 180 images per class.

### Overall Performance Comparison

| Evaluation Metric | CNN Version 1 | CNN Version 3 | MobileNetV2 |
|---|---:|---:|---:|
| Test Accuracy | 77.22% | 75.28% | **96.39%** |
| Precision (Macro) | 76.80% | 76.17% | **96.40%** |
| Recall (Macro) | 77.22% | 75.28% | **96.39%** |
| F1-Score (Macro) | 76.83% | 74.91% | **96.38%** |
| Test Loss | 1.3804 | 0.7485 | **0.1612** |
| Parameters | 4,288,454 | 424,006 | 2,422,726 |
| Inference Time/Image | **1.28 ms** | 1.45 ms | 3.13 ms |

*Note: Inference times are based on the recorded evaluation measurements and may vary depending on hardware and execution environment.*

### Key Findings

- **MobileNetV2 achieved the highest test accuracy of 96.39%**, outperforming both custom CNN architectures.
- MobileNetV2 also achieved the highest macro precision, recall, and F1-score.
- CNN Version 3 has the fewest parameters, making it the smallest architecture among the three.
- CNN Version 1 recorded the lowest average inference time per image.
- MobileNetV2 demonstrated particularly strong classification performance across all six insect categories.

### MobileNetV2 Class-Wise Performance

| Insect Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Ant | 95.65% | 97.78% | 96.70% |
| Bee | 94.09% | 97.22% | 95.63% |
| Beetle | 95.35% | 91.11% | 93.18% |
| Butterfly | 99.44% | 98.33% | 98.88% |
| Grasshopper | 96.61% | 95.00% | 95.80% |
| Ladybug | 97.27% | 98.89% | 98.07% |

**Best-performing class:** Butterfly, with an F1-score of 98.88%.

**Most challenging class:** Beetle, with a recall of 91.11%.

## 🖥️ Application Pages

The Streamlit application contains the following pages:

| Page | Description |
|---|---|
| Dashboard Overview | Displays project information and model performance summaries. |
| Live Classification | Allows users to upload insect images and generate predictions. |
| Model Comparison | Compares the performance of all three trained models. |
| Training Analysis | Visualizes training and validation accuracy and loss. |
| Dataset Explorer | Explores dataset information, class distribution, and sample images. |
| Model Evaluation | Displays confusion matrices, classification reports, and evaluation metrics. |
| Prediction History | Tracks previous predictions and provides prediction analytics. |
| Model Explainability | Generates Grad-CAM visualizations for supported CNN models. |

## 🔍 Grad-CAM Explainability

InsectAI incorporates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to provide visual explanations for model predictions.

Grad-CAM highlights image regions that contribute to a model's classification decision.

The explainability module provides:

- Original uploaded image.
- Grad-CAM heatmap.
- Heatmap overlay on the original image.
- Visual interpretation of important image regions.

Grad-CAM is currently implemented for CNN Version 1 and CNN Version 3.

This feature helps make model predictions more interpretable by providing visual insights into the regions influencing classification.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | Pillow, NumPy |
| Data Analysis | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Machine Learning Metrics | Scikit-learn |
| Web Application | Streamlit |
| Model Explainability | Grad-CAM |
| Development Environment | Kaggle Notebooks, Visual Studio Code |
| Version Control | Git, GitHub |

## 📁 Project Structure

```text
InsectAI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── cnn_version_1.keras
│   ├── cnn_version_3.keras
│   └── mobilenetv2_insect_classifier.keras
│
├── artifacts/
│   ├── model_metrics.json
│   ├── classification_reports.json
│   ├── confusion_matrices.json
│   ├── dataset_metadata.json
│   ├── training_history.json
│   ├── model_metadata.json
│   ├── roc_pr_curves.json
│   ├── prediction_history.csv
│   │
│   └── sample_images/
│       ├── ant/
│       ├── bee/
│       ├── beetle/
│       ├── butterfly/
│       ├── grasshopper/
│       └── ladybug/
│
├── utils/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── preprocessing.py
│   ├── prediction.py
│   ├── data_loader.py
│   └── explainability.py
│
├── pages/
│   ├── classification.py
│   ├── comparison.py
│   ├── training_analysis.py
│   ├── dataset_explorer.py
│   ├── model_evaluation.py
│   ├── prediction_history.py
│   └── model_explainability.py
│
├── assets/
│
├── notebooks/
│
└── inspect_metrics.py
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Khubaib-Muhammad/InsectAI.git
```

```bash
cd InsectAI
```

### 2. Create a Virtual Environment

```bash
python -m venv myenv
```

Activate it on Windows:

```powershell
myenv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Project Files

Ensure the following directories are available:

- `models/` containing the trained `.keras` model files.
- `artifacts/` containing the required JSON evaluation files and supporting artifacts.
- `utils/` containing the application utility modules.
- `pages/` containing the Streamlit application pages.

## 🚀 Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

You can also access the deployed application here:

🔗 [Launch InsectAI](https://insectai-qnsyy75tjbe2pkvbv9mhfe.streamlit.app/model_explainability)

## 🔮 Future Improvements

Potential future enhancements include:

- Adding more insect categories.
- Improving custom CNN architectures.
- Optimizing inference speed.
- Extending Grad-CAM support to MobileNetV2.
- Adding more explainability techniques.
- Implementing real-time insect detection using object detection models.
- Improving model generalization with additional training data.
- Deploying the application with scalable cloud infrastructure.

---

## 👨‍💻 Author

**Khubaib Muhammad**

AI Undergraduate | Machine Learning & Deep Learning Enthusiast

GitHub: [Khubaib-Muhammad](https://github.com/Khubaib-Muhammad)

---

⭐ If you find this project interesting, consider giving the repository a star!

