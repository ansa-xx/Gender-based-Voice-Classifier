import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import torchaudio
import whisper
import numpy as np
from torchvision import transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
img_size = 128
class_names = ['female', 'male']  # Must match training order

# Load trained CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load model
model = SimpleCNN(num_classes=2).to(device)
model.load_state_dict(torch.load("best_gender_cnn.pth", map_location=device))
model.eval()

# Load whisper model
whisper_model = whisper.load_model("tiny")

# Image transform (must match training)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def audio_to_image_tensor(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    # Resample to 16kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio = waveform.squeeze().float()
    mel = whisper.log_mel_spectrogram(audio)

    # Convert to numpy and scale to 0–255
    mel_np = mel.cpu().numpy()
    mel_img = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-6) * 255
    mel_img = mel_img.astype(np.uint8)

    # Convert to PIL image and resize
    img = Image.fromarray(mel_img)
    img = img.resize((img_size, img_size))
    img = img.convert("RGB")  # Ensure 3 channels

    # Apply same transform as training
    img_tensor = transform(img)  # Shape: [3, 128, 128]
    return img_tensor

def predict_gender(audio_path):
    input_tensor = audio_to_image_tensor(audio_path).unsqueeze(0).to(device)  # Shape: [1, 3, 128, 128]

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0, pred_idx].item()

    predicted_label = class_names[pred_idx]
    return predicted_label, confidence

# CLI usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_gender.py path/to/audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.isfile(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    label, confidence = predict_gender(audio_path)
    print(f"Prediction: {label.capitalize()} ({confidence * 100:.2f}% confidence)")

