import os
import librosa

audio_folder = "data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac"
files = os.listdir(audio_folder)

sample_file = os.path.join(audio_folder, files[0])
y, sr = librosa.load(sample_file, sr=None)

print("Loaded file:", sample_file)
print("Sample rate:", sr)
print("Audio shape:", y.shape)
print("Duration (sec):", len(y) / sr)