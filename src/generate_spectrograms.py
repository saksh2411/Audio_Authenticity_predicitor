import os
import sys
import pandas as pd
import librosa
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from config import (
    SAMPLE_RATE,
    TRAIN_PREPROCESSED_CSV,
    VAL_PREPROCESSED_CSV,
    TRAIN_SPECTROGRAM_DIR,
    VAL_SPECTROGRAM_DIR,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMAX
)

os.makedirs(TRAIN_SPECTROGRAM_DIR, exist_ok=True)
os.makedirs(VAL_SPECTROGRAM_DIR, exist_ok=True)


def create_log_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=FMAX
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


def process_csv(input_csv, output_dir, output_csv):
    if not os.path.exists(input_csv):
        print(f"CSV not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)

    if df.empty:
        print(f"CSV is empty: {input_csv}")
        return

    if "processed_filepath" not in df.columns or "file_id" not in df.columns:
        print("CSV must contain 'processed_filepath' and 'file_id' columns")
        print("Found columns:", df.columns.tolist())
        return

    spectrogram_paths = []
    status_list = []
    error_list = []

    print(f"\nGenerating spectrograms for: {input_csv}")
    print(f"Total rows: {len(df)}")

    for idx, row in df.iterrows():
        audio_path = str(row["processed_filepath"]).strip()
        file_id = str(row["file_id"]).strip()
        output_path = os.path.join(output_dir, file_id + ".npy")

        print(f"[{idx + 1}/{len(df)}] {audio_path}")

        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Processed audio not found: {audio_path}")

            log_mel = create_log_mel_spectrogram(audio_path)
            np.save(output_path, log_mel)

            spectrogram_paths.append(output_path)
            status_list.append("ok")
            error_list.append("")

        except Exception as e:
            spectrogram_paths.append("")
            status_list.append("failed")
            error_list.append(str(e))
            print(f"   Failed: {e}")

    df["spectrogram_path"] = spectrogram_paths
    df["spectrogram_status"] = status_list
    df["spectrogram_error"] = error_list

    df.to_csv(output_csv, index=False)

    print(f"\nSaved spectrogram metadata to: {output_csv}")
    print(df["spectrogram_status"].value_counts())


if __name__ == "__main__":
    train_output_csv = os.path.join(
        os.path.dirname(TRAIN_PREPROCESSED_CSV), "train_spectrograms.csv"
    )
    val_output_csv = os.path.join(
        os.path.dirname(VAL_PREPROCESSED_CSV), "val_spectrograms.csv"
    )

    process_csv(TRAIN_PREPROCESSED_CSV, TRAIN_SPECTROGRAM_DIR, train_output_csv)
    process_csv(VAL_PREPROCESSED_CSV, VAL_SPECTROGRAM_DIR, val_output_csv)

    print("\nSpectrogram generation finished")