import os
import pandas as pd

BASE_PATH = "data/raw/ASVspoof2019_LA"
PROTOCOL_PATH = os.path.join(BASE_PATH, "ASVspoof2019_LA_cm_protocols")
OUTPUT_PATH = "data/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def parse_protocol_file(protocol_file, audio_folder, split_name, output_csv):
    rows = []

    with open(protocol_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 4:
                continue

            # Adjust based on your actual file format
            speaker_id = parts[0]
            file_id = parts[1]

            # Try to infer system/attack id and label from the last columns
            if parts[-1] in ["bonafide", "spoof"]:
                label = parts[-1]
                attack_id = parts[-2] if len(parts) >= 5 else "-"
            else:
                continue

            label_num = 0 if label == "bonafide" else 1

            audio_path = os.path.join(audio_folder, file_id + ".flac")

            rows.append({
                "speaker_id": speaker_id,
                "file_id": file_id,
                "filepath": audio_path,
                "label": label,
                "label_num": label_num,
                "split": split_name,
                "attack_id": attack_id
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"{split_name} metadata saved to: {output_csv}")
    print(df.head())
    print("Total rows:", len(df))


if __name__ == "__main__":
    train_protocol = os.path.join(PROTOCOL_PATH, "ASVspoof2019.LA.cm.train.trn.txt")
    dev_protocol = os.path.join(PROTOCOL_PATH, "ASVspoof2019.LA.cm.dev.trl.txt")

    train_audio_folder = os.path.join(BASE_PATH, "ASVspoof2019_LA_train", "flac")
    dev_audio_folder = os.path.join(BASE_PATH, "ASVspoof2019_LA_dev", "flac")

    train_output = os.path.join(OUTPUT_PATH, "train.csv")
    dev_output = os.path.join(OUTPUT_PATH, "val.csv")

    parse_protocol_file(train_protocol, train_audio_folder, "train", train_output)
    parse_protocol_file(dev_protocol, dev_audio_folder, "val", dev_output)