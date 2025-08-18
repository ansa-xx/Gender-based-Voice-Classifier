import os
from predict_gender import predict_gender  # Ensure predict_gender.py is in the same folder or in PYTHONPATH

# Path to directory containing audio files
test_dir = "your directory"  # Change to your directory

# Filter supported audio files
audio_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.wav', '.mp3'))]
results = []
print(f"Found {len(audio_files)} audio files in: {test_dir}")

for file_name in audio_files:
    file_path = os.path.join(test_dir, file_name)
    try:
        label, confidence = predict_gender(file_path)
        results.append((file_name, label, confidence))
        print(f"{file_name}: {label.capitalize()} ({confidence * 100:.2f}%)")
    except Exception as e:
        print(f"Failed on {file_name}: {e}")

# Save results to a text file
output_path = "prediction_results.txt"
with open(output_path, "w") as f:
    for file_name, label, confidence in results:
        f.write(f"{file_name}\t{label}\t{confidence:.4f}\n")

print(f"\n Batch prediction complete. Results saved to: {output_path}")
