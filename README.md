# 🧠 MNIST Digit Classification using TensorFlow/Keras

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25--98%25-brightgreen.svg)

A simple yet effective deep learning project that classifies handwritten digits (0–9) using a fully connected neural network built with TensorFlow/Keras.

---

## 🚀 Overview

This project demonstrates how to build, train, and evaluate a neural network on the MNIST dataset. It also includes visualization of predictions and misclassified examples for model interpretability.

---

## 📊 Dataset

* **Dataset:** MNIST Handwritten Digits
* **Training Samples:** 60,000
* **Test Samples:** 10,000
* **Image Size:** 28 × 28 grayscale

---

## 🏗️ Model Architecture

A simple feedforward neural network:

* Input Layer: 784 neurons (flattened 28×28 image)
* Hidden Layer 1: 128 neurons (ReLU)
* Hidden Layer 2: 64 neurons (ReLU)
* Output Layer: 10 neurons (Softmax)

---

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib

---

## ▶️ Usage

Run the training script:

```bash
python train.py
```

---

## 📈 Training Details

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Epochs: 5
* Batch Size: 32

---

## ✅ Results

* Achieves ~97–98% accuracy on the test dataset
* Efficient training with low computational cost

---

## 🔍 Visualizations

### 🧾 Sample Predictions

Displays model predictions vs actual labels for training data.

### ❌ Misclassified Examples

Highlights incorrect predictions to analyze model weaknesses.

---

## 📌 Key Learnings

* Data preprocessing (normalization, reshaping)
* Building neural networks with Keras
* Model evaluation and visualization
* Debugging misclassifications

---

## 🔧 Future Improvements

* Add Convolutional Neural Networks (CNNs)
* Hyperparameter tuning
* Model checkpointing & logging
* Deployment using FastAPI or Streamlit
* Convert to production-ready pipeline

---

## ⭐ Acknowledgements

* TensorFlow/Keras documentation
* MNIST dataset creators

---

## 💡 Note

This project is a foundational deep learning implementation and serves as a stepping stone toward more advanced architectures like CNNs and transformer-based vision models.

---
