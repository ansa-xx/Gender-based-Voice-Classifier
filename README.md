# 🎙️ Gender Classification from Voice using Whisper & CNN

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)
![Whisper](https://img.shields.io/badge/Whisper-OpenAI-blueviolet)

This project performs **gender classification (Male / Female)** using audio samples. It leverages **OpenAI's Whisper** model to extract `log-mel spectrograms`, which are then fed into a **Convolutional Neural Network (CNN)** for classification.

### 🔍 Use-Cases:
- 🎯 Content recommendation
- 🔐 Voice-based parental controls
- 📊 Audience analysis
- 🔍 Voice-based personalization systems

---

## 📁 Project Structure

gender_classifier/
├── audio/ # Raw audio files organized by gender
│ ├── male/
│ ├── female/
│ └── kid/
├── data/ # Generated mel-spectrogram images (128x128)
│ ├── male/
│ ├── female/
│ └── kid/
├── save_mel_images.py # Converts audio to mel-spectrogram PNGs
├── train_gender.py # Training script for CNN model
├── predict_gender.py # Predict gender for a single audio file
├── batch_predict.py # Predict gender for a folder of audio files
└── best_gender_cnn.pth # Trained model weights (PyTorch)

---

## ⚙️ How to Run

> ✅ **Important:** Update folder and file paths in scripts based on your local setup or environment.

### 🔹 Step 1: Preprocess Audio to Mel-Spectrograms
This will convert `.wav` or `.mp3` files from `audio/` into 128x128 mel-spectrogram images stored in the `data/` directory.
```bash
python save_mel_images.py
```

### 🔹 Step 2: Train the Model
This script trains a CNN model and saves the trained weights as `best_gender_cnn.pth`.
```bash
python train_gender.py
```

### 🔹 Step 3: Predict Gender of a Single Audio File
This script classifies a single audio file
```bash
python predict_gender.py --input path_to_audio.wav
```

### 🔹 Step 4: Batch Predict
This script classifies all audio files in a folder.
```bash
python batch_predict.py --input path_to_audio.wav
```
