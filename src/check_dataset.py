import os

base_path = "data/raw/ASVspoof2019_LA"

folders = [
    "ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019_LA_train",
    "ASVspoof2019_LA_dev",
    "ASVspoof2019_LA_eval"
]

for folder in folders:
    full_path = os.path.join(base_path, folder)
    print(f"{folder}: {'FOUND' if os.path.exists(full_path) else 'MISSING'}")