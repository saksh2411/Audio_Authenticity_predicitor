import os
import sys
import pandas as pd
import librosa
import soundfile as sf
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from config import (
    SAMPLE_RATE,
    DURATION,
    TARGET_LENGTH,
    TRAIN_CSV,
    VAL_CSV,
    TRAIN_OUTPUT_DIR,
    VAL_OUTPUT_DIR
)

os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(VAL_OUTPUT_DIR, exist_ok=True)


def preprocess_audio_file(input_path, output_path):
    try:
        if not os.path.exists(input_path):
            return False, f"Input file not found: {input_path}"

        y, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)

        if y is None or len(y) == 0:
            return False, f"Empty audio: {input_path}"

        if len(y) > TARGET_LENGTH:
            y = y[:TARGET_LENGTH]
        elif len(y) < TARGET_LENGTH:
            y = np.pad(y, (0, TARGET_LENGTH - len(y)))

        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        sf.write(output_path, y, SAMPLE_RATE)
        return True, "ok"

    except Exception as e:
        return False, str(e)


def process_csv(input_csv, output_dir, output_csv):
    if not os.path.exists(input_csv):
        print(f"CSV not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)

    if df.empty:
        print(f"CSV is empty: {input_csv}")
        return

    if "filepath" not in df.columns or "file_id" not in df.columns:
        print("CSV must contain 'filepath' and 'file_id' columns")
        print("Found columns:", df.columns.tolist())
        return

    processed_paths = []
    status_list = []
    error_list = []

    print(f"\nProcessing file: {input_csv}")
    print(f"Total rows: {len(df)}")
    print(f"Output directory: {output_dir}\n")

    for idx, row in df.iterrows():
        input_path = str(row["filepath"]).strip()
        file_id = str(row["file_id"]).strip()

        output_path = os.path.join(output_dir, file_id + ".wav")

        print(f"[{idx + 1}/{len(df)}] {input_path}")

        success, message = preprocess_audio_file(input_path, output_path)

        if success:
            processed_paths.append(output_path)
            status_list.append("ok")
            error_list.append("")
        else:
            processed_paths.append("")
            status_list.append("failed")
            error_list.append(message)
            print(f"   Failed: {message}")

    df["processed_filepath"] = processed_paths
    df["preprocess_status"] = status_list
    df["error_message"] = error_list

    df.to_csv(output_csv, index=False)

    print(f"\nSaved processed metadata to: {output_csv}")
    print(df["preprocess_status"].value_counts())

    failed_df = df[df["preprocess_status"] == "failed"]
    if not failed_df.empty:
        print("\nSome files failed. First few errors:")
        print(failed_df[["filepath", "error_message"]].head())


if __name__ == "__main__":
    print("Script started")
    print("TRAIN_CSV:", TRAIN_CSV)
    print("VAL_CSV:", VAL_CSV)
    print("TRAIN_OUTPUT_DIR:", TRAIN_OUTPUT_DIR)
    print("VAL_OUTPUT_DIR:", VAL_OUTPUT_DIR)

    train_output_csv = os.path.join(
        os.path.dirname(TRAIN_CSV), "train_preprocessed.csv"
    )
    val_output_csv = os.path.join(
        os.path.dirname(VAL_CSV), "val_preprocessed.csv"
    )

    process_csv(TRAIN_CSV, TRAIN_OUTPUT_DIR, train_output_csv)
    process_csv(VAL_CSV, VAL_OUTPUT_DIR, val_output_csv)

    print("\nPreprocessing finished")