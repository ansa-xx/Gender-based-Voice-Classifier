import os
import whisper
import torchaudio
from PIL import Image
import numpy as np
from tqdm import tqdm

# Load Whisper model (tiny is fast and good enough for mel)
model = whisper.load_model("tiny")

# Allowed labels
ALLOWED_LABELS = {"male", "female"}

def save_mel_image(audio_path, label, save_dir):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to 16kHz
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio = waveform.squeeze().float()
    mel = whisper.log_mel_spectrogram(audio)

    mel_np = mel.cpu().numpy()
    mel_img = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min()) * 255
    mel_img = mel_img.astype(np.uint8)

    # Convert to PIL and resize to 128x128
    img = Image.fromarray(mel_img)
    img = img.resize((128, 128))  # Match CNN input
    img = img.convert("RGB")  # Ensure 3 channels

    # Save to correct folder
    os.makedirs(os.path.join(save_dir, label), exist_ok=True)
    save_name = os.path.basename(audio_path).replace('.mp3', '.png').replace('.wav', '.png')
    save_path = os.path.join(save_dir, label, save_name)
    img.save(save_path)

def process_dataset(input_dir="audio", output_dir="data"):
    labels = os.listdir(input_dir)
    for label in labels:
        if label not in ALLOWED_LABELS:
            print(f"Skipping unknown label: {label}")
            continue

        label_path = os.path.join(input_dir, label)
        if not os.path.isdir(label_path):
            continue

        for filename in tqdm(os.listdir(label_path), desc=f"Processing '{label}'"):
            if filename.endswith(".mp3") or filename.endswith(".wav"):
                audio_path = os.path.join(label_path, filename)
                try:
                    save_mel_image(audio_path, label, output_dir)
                except Exception as e:
                    print(f" Error processing {audio_path}: {e}")

if __name__ == "__main__":
    process_dataset()
    print("Done generating mel spectrogram images.")

