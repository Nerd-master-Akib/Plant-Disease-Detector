# Plant-Disease-Detector
# 🌱 Plant Disease Detection using CNN (PyTorch + Streamlit)

## 📌 Project Overview

This project is a Convolutional Neural Network (CNN) based Plant Disease Detection system built using PyTorch and deployed with Streamlit.

The model classifies leaf images into different disease categories and provides real-time predictions through a web interface.


## 🚀 Features

- Upload leaf images through a web UI
- Predict plant disease using a trained CNN model
- Show prediction confidence
- Display class-wise probabilities
- Simple and interactive Streamlit interface


## 📊 Dataset

This project uses the **PlantVillage Dataset**.

Only 5 classes were selected:

- Potato Early Blight  
- Potato Healthy  
- Tomato Early Blight  
- Tomato Late Blight  
- Tomato Healthy  

Total dataset size: ~5652 images


## 🧠 Model Architecture

A custom CNN was used with the following structure:

- 3 Convolutional layers
- ReLU activation
- MaxPooling layers
- Fully connected layers

Input image size: **224 × 224**


## 📈 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 96.87% |
| Validation Accuracy | 91.26% |
| Classes | 5 |
| Image Size | 224×224 |


## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- Pillow
- Pandas
📂 Project Structure
plant-disease-detection-cnn/
│
├── app.py
├── model.py
├── train.py
├── predict.py
├── requirements.txt


---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
Example Output
Input: Leaf image
Output: Predicted disease class
Confidence score displayed in % 


What I Learned
Building CNNs with PyTorch
Image preprocessing and augmentation
Training and evaluating deep learning models
Model saving and loading
Deploying ML models using Streamlit


Future Improvements
Use Transfer Learning (ResNet / MobileNet)
Improve accuracy on real-world images
Deploy online (Streamlit Cloud / HuggingFace)
Add treatment recommendations for diseases 


Note : 
## Dataset

The PlantVillage dataset was used for training.

Dataset is not included in this repository due to size limitations.

You can download it from Kaggle or PlantVillage sources.

## Trained Model

The trained model file is not included due to size limits.

To generate it:
1. Download dataset
2. Run train.py

👨‍💻 Author

Built by Akib 🚀
First Computer Vision Project using Deep Learning 
  

## 📂 Project Structure

