import pandas as pd

df = pd.read_csv("data/features/train_pause_prosody_features.csv")

print("Columns:")
print(df.columns.tolist())

print("\nFeature status counts:")
print(df["feature_status"].value_counts())

print("\nSample rows:")
print(df[[
    "file_id",
    "label",
    "speech_ratio",
    "num_pauses",
    "mean_pause_duration",
    "micropause_rate",
    "macropause_rate",
    "mean_f0",
    "std_f0"
]].head())