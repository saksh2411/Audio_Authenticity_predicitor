import os
import sys
import numpy as np
import pandas as pd
import librosa

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from config import (
    SAMPLE_RATE,
    TRAIN_PREPROCESSED_CSV,
    VAL_PREPROCESSED_CSV,
    TRAIN_FEATURE_CSV,
    VAL_FEATURE_CSV,
    FEATURE_DIR
)

os.makedirs(FEATURE_DIR, exist_ok=True)


def get_pause_and_speech_segments(y, sr, top_db=30):
    intervals = librosa.effects.split(y, top_db=top_db)
    total_duration = len(y) / sr

    speech_segments = []
    pause_segments = []

    for start, end in intervals:
        speech_segments.append((start / sr, end / sr))

    if len(speech_segments) == 0:
        return [], [(0.0, total_duration)]

    if speech_segments[0][0] > 0:
        pause_segments.append((0.0, speech_segments[0][0]))

    for i in range(len(speech_segments) - 1):
        pause_start = speech_segments[i][1]
        pause_end = speech_segments[i + 1][0]
        if pause_end > pause_start:
            pause_segments.append((pause_start, pause_end))

    if speech_segments[-1][1] < total_duration:
        pause_segments.append((speech_segments[-1][1], total_duration))

    return speech_segments, pause_segments


def safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else 0.0


def safe_std(values):
    return float(np.std(values)) if len(values) > 0 else 0.0


def extract_features(audio_path, sr=SAMPLE_RATE):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    total_duration = len(y) / sr

    speech_segments, pause_segments = get_pause_and_speech_segments(y, sr, top_db=30)

    speech_durations = [end - start for start, end in speech_segments]
    pause_durations = [end - start for start, end in pause_segments]

    total_speech_time = sum(speech_durations)
    speech_ratio = total_speech_time / total_duration if total_duration > 0 else 0.0

    micropauses = [p for p in pause_durations if 0.1 <= p < 0.5]
    macropauses = [p for p in pause_durations if p >= 0.5]

    micropause_rate = len(micropauses) / total_duration if total_duration > 0 else 0.0
    macropause_rate = len(macropauses) / total_duration if total_duration > 0 else 0.0

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )

    voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    features = {
        "speech_ratio": speech_ratio,
        "num_pauses": len(pause_durations),
        "mean_pause_duration": safe_mean(pause_durations),
        "std_pause_duration": safe_std(pause_durations),
        "mean_speech_segment_duration": safe_mean(speech_durations),
        "std_speech_segment_duration": safe_std(speech_durations),
        "micropause_rate": micropause_rate,
        "macropause_rate": macropause_rate,
        "mean_f0": safe_mean(voiced_f0),
        "std_f0": safe_std(voiced_f0),
        "voiced_ratio": len(voiced_f0) / len(f0) if f0 is not None and len(f0) > 0 else 0.0,
        "rms_mean": safe_mean(rms),
        "rms_std": safe_std(rms),
        "zcr_mean": safe_mean(zcr),
        "zcr_std": safe_std(zcr),
        "audio_duration": total_duration
    }

    return features


def process_csv(input_csv, output_csv):
    if not os.path.exists(input_csv):
        print(f"CSV not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)

    if "processed_filepath" not in df.columns:
        print(f"'processed_filepath' column missing in {input_csv}")
        return

    records = []

    print(f"\nProcessing: {input_csv}")
    print(f"Total rows: {len(df)}")

    for idx, row in df.iterrows():
        audio_path = str(row["processed_filepath"]).strip()

        print(f"[{idx + 1}/{len(df)}] {audio_path}")

        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            feats = extract_features(audio_path)

            record = row.to_dict()
            record.update(feats)
            record["feature_status"] = "ok"
            record["feature_error"] = ""

        except Exception as e:
            record = row.to_dict()
            record["feature_status"] = "failed"
            record["feature_error"] = str(e)

        records.append(record)

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)

    print(f"\nSaved features to: {output_csv}")
    if "feature_status" in out_df.columns:
        print(out_df["feature_status"].value_counts())


if __name__ == "__main__":
    process_csv(TRAIN_PREPROCESSED_CSV, TRAIN_FEATURE_CSV)
    process_csv(VAL_PREPROCESSED_CSV, VAL_FEATURE_CSV)

    print("\nPause and prosody feature extraction complete")