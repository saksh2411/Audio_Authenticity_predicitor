import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/train_spectrograms.csv")
df = df[df["spectrogram_status"] == "ok"]

for i in range(5):
    path = df.iloc[i]["spectrogram_path"]
    spec = np.load(path)
    print(path, spec.shape)