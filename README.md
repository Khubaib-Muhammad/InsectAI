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
- Trainable parameters: 4,288,454

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
- Trainable parameters: 424,006

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
- Total trainable parameters: 2,422,726

## 📊 Model Performance

The custom CNN models were evaluated on the held-out test dataset.

| Evaluation Metric | CNN Version 1 | CNN Version 3 |
|---|---:|---:|
| Test Accuracy | 78.06% | 73.61% |
| Precision | 78.31% | 74.35% |
| Recall | 78.06% | 73.61% |
| F1-Score | 77.91% | 73.06% |
| Test Loss | 1.2121 | 0.7905 |
| Parameters | 4,288,454 | 424,006 |
| Inference Time/Image | 1.31 ms | 1.46 ms |

**Note:** MobileNetV2 achieved 96.24% validation accuracy, but its initial test evaluation produced 4.81% accuracy. This discrepancy is under investigation, so its results should not be treated as a reliable performance comparison until the evaluation pipeline is verified.

The application provides additional model metrics and visual comparisons.

## 🖥️ Application Pages

The Streamlit application is organized into eight interactive pages.

### 1. Dashboard Overview
Provides an overview of the InsectAI platform, dataset, supported classes, and available models.

### 2. Live Classification
Allows users to upload an insect image and receive:
- Predicted insect category.
- Prediction confidence.
- Class-wise probability distribution.
- Selected model information.

### 3. Model Comparison
Compares the three deep learning architectures using performance metrics, visualizations, and model characteristics.

### 4. Training Analysis
Displays:
- Training accuracy.
- Validation accuracy.
- Training loss.
- Validation loss.
- Best validation accuracy.
- Best validation loss.
- Epoch-wise training progress.

### 5. Dataset Explorer
Provides dataset statistics, class distributions, and sample images from each insect category.

### 6. Model Evaluation
Includes:
- Overall model performance.
- Confusion matrices.
- Classification reports.
- Class-wise precision, recall, and F1-score.
- Macro and weighted averages.

### 7. Prediction History & Analytics
Maintains a record of previous classification results.

Stored information includes:
- Prediction timestamp.
- Uploaded image filename.
- Selected model.
- Predicted insect category.
- Prediction confidence.

Users can filter, analyze, and export prediction records.

### 8. Model Explainability (Grad-CAM)
Uses Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize which image regions influence CNN predictions.

The page displays:
- Original uploaded image.
- Grad-CAM heatmap.
- Heatmap overlay.
- Predicted class and confidence.
- Class-wise prediction probabilities.

Grad-CAM is currently configured for CNN Version 1 and CNN Version 3.

## 🔍 Grad-CAM Explainability

A major feature of InsectAI is the integration of explainable AI.

Instead of providing only a predicted class, Grad-CAM helps visualize the image regions that contribute to a CNN's prediction.

The implementation:
1. Extracts feature maps from the selected convolutional layer.
2. Calculates gradients with respect to the predicted class.
3. Computes channel importance weights.
4. Generates a class activation heatmap.
5. Resizes the heatmap to match the original image.
6. Combines the heatmap with the uploaded image.

This provides a visual interpretation of the model's classification behavior.

**Note:** Grad-CAM visualizations are approximate explanations of model behavior, not proof that a prediction is correct.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | Neural network development |
| MobileNetV2 | Transfer learning architecture |
| NumPy | Numerical computations |
| Pandas | Data processing and analysis |
| Matplotlib | Visualization and heatmap generation |
| Pillow (PIL) | Image processing |
| Scikit-learn | Model evaluation metrics |
| Streamlit | Interactive web application |
| Plotly | Interactive charts and visualizations |
| Git & GitHub | Version control and project hosting |

## 📁 Project Structure

```text
InsectAI/
│
├── app.py
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
├── notebooks/
│   └── training_notebooks.ipynb
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
├── utils/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── preprocessing.py
│   ├── prediction.py
│   ├── data_loader.py
│   └── explainability.py
│
├── assets/
│
├── inspect_metrics.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

Follow these steps to run InsectAI locally.

### Prerequisites

- Python 3.10 or compatible version supported by the installed TensorFlow release.
- pip package manager.
- Git.
- Sufficient memory and storage for loading the trained models.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/InsectAI.git
```

### 2. Navigate to the Project Directory

```bash
cd InsectAI
```

### 3. Create a Virtual Environment

```bash
python -m venv myenv
```

### 4. Activate the Virtual Environment

**Windows:**

```powershell
myenv\Scripts\activate
```

**Linux / macOS:**

```bash
source myenv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## 🚀 Live Demo

Explore the deployed InsectAI application:

**[Launch InsectAI on Streamlit](https://insectai-qnsyy75tjbe2pkvbv9mhfe.streamlit.app/model_explainability)**

Upload an insect image, explore classification predictions, compare models, and visualize Grad-CAM explanations.

## 🔮 Future Improvements

Potential future enhancements include:

- Investigating and improving MobileNetV2 test performance.
- Improving classification accuracy through hyperparameter optimization.
- Adding additional insect categories.
- Integrating additional explainability methods such as SHAP.
- Improving model robustness across different lighting conditions and backgrounds.
- Adding prediction confidence calibration.
- Supporting batch image classification.
- Improving deployment performance and model loading efficiency.

## 👨‍💻 Author

**Khubaib Muhammad**

Computer Science / Artificial Intelligence Student

GitHub: [Khubaib-Muhammad](https://github.com/Khubaib-Muhammad)

---

## 📄 License

This project is intended for educational and research purposes.

If you plan to distribute or reuse this project, add an appropriate open-source license to the repository.

---

⭐ If you find this project interesting, consider giving the repository a star!
