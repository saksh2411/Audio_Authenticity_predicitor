import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/train_spectrograms.csv")
sample_path = df[df["spectrogram_status"] == "ok"].iloc[0]["spectrogram_path"]

spec = np.load(sample_path)

print("Spectrogram file:", sample_path)
print("Shape:", spec.shape)
print("Min value:", spec.min())
print("Max value:", spec.max()) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

df = pd.read_csv("data/processed/train_spectrograms.csv")
sample_path = df[df["spectrogram_status"] == "ok"].iloc[0]["spectrogram_path"]

spec = np.load(sample_path)

plt.figure(figsize=(10, 4))
librosa.display.specshow(spec, sr=16000, hop_length=256, x_axis="time", y_axis="mel", fmax=8000)
plt.colorbar(format="%+2.0f dB")
plt.title("Log-Mel Spectrogram")
plt.tight_layout()
plt.show()