import pandas as pd
import librosa

df = pd.read_csv("data/processed/train_preprocessed.csv")

sample_path = df[df["preprocess_status"] == "ok"].iloc[0]["processed_filepath"]
y, sr = librosa.load(sample_path, sr=None)

print("Processed file:", sample_path)
print("Sample rate:", sr)
print("Length in samples:", len(y))
print("Duration in seconds:", len(y) / sr)
print("Min value:", y.min())
print("Max value:", y.max())

print("Script started")
print("Training CSV:", TRAIN_CSV)
print("Validation CSV:", VAL_CSV)
print("Processing:", input_path)