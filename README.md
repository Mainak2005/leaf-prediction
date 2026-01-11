# 🍀 Potato Leaf Disease Classification using CNN, FastAPI & Streamlit

This project implements a **Convolutional Neural Network (CNN)** to classify potato leaf diseases from images. It provides a **FastAPI backend** for model inference and a **Streamlit frontend** for easy image upload and prediction display.

---

## Features

- CNN model for multi-class potato leaf disease classification
- FastAPI backend for handling image predictions
- Streamlit frontend for uploading images and viewing results
- Displays **predicted disease class** with **confidence**
- Save and reuse trained model (`.h5`) and class labels (`.json`)

---

## Project Structure
Potato_Leaf_Project/
├── Potato Leaf Disease/ # Dataset (not uploaded to GitHub)
├── .venv/ # Python virtual environment (not uploaded)
├── app.py # FastAPI backend
├── streamlit_app.py # Streamlit frontend
├── potato_leaf_cnn.h5 # Trained CNN model
├── class_labels.json # Class labels mapping
├── requirements.txt # Project dependencies
├── README.md # Project documentation

The dataset folder should contain subfolders for each disease class.

Each subfolder contains images of potato leaves for that class.

Example structure:

Potato Leaf Disease/
├── Early_Blight/
├── Late_Blight/
├── Healthy/
├── Class_4/
├── Class_5/
├── Class_6/
├── Class_7/

Dependencies

Python 3.12+
TensorFlow / Keras
FastAP
Uvicorn
Streamlit
Requests
Pillow / Numpy / Matplotlib





