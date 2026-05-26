import pandas as pd
import os

df = pd.read_csv("data/processed/train.csv")

missing_files = []
for path in df["filepath"]:
    if not os.path.exists(path):
        missing_files.append(path)

print("Total rows:", len(df))
print("Missing files:", len(missing_files))

if missing_files:
    print("First few missing paths:")
    for p in missing_files[:5]:
        print(p)
else:
    print("All file paths are valid.")

import pandas as pd

df = pd.read_csv("data/processed/train.csv")
print(df["label"].value_counts())
print(df["label_num"].value_counts())