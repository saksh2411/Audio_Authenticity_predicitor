import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from config import (
    VAL_SPECTROGRAM_CSV,
    VAL_FEATURE_CSV,
    MODEL_DIR,
    CLASSICAL_MODEL_DIR,
    FUSION_MODEL_DIR,
    FUSION_REPORT_DIR
)

os.makedirs(FUSION_MODEL_DIR, exist_ok=True)
os.makedirs(FUSION_REPORT_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "speech_ratio",
    "num_pauses",
    "mean_pause_duration",
    "std_pause_duration",
    "mean_speech_segment_duration",
    "std_speech_segment_duration",
    "micropause_rate",
    "macropause_rate",
    "mean_f0",
    "std_f0",
    "voiced_ratio",
    "rms_mean",
    "rms_std",
    "zcr_mean",
    "zcr_std",
    "audio_duration"
]

BATCH_SIZE = 32


def load_cnn_metadata(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["spectrogram_status"] == "ok"].copy()
    return df[["file_id", "label_num", "spectrogram_path"]].reset_index(drop=True)


def load_feature_validation_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["feature_status"] == "ok"].copy()

    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    return df[["file_id", "label_num"] + FEATURE_COLUMNS].reset_index(drop=True)


def predict_cnn_in_batches(model, spectrogram_paths, batch_size=32):
    probs = []

    for start in range(0, len(spectrogram_paths), batch_size):
        batch_paths = spectrogram_paths[start:start + batch_size]
        batch_specs = []

        for path in batch_paths:
            spec = np.load(path)
            spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
            spec = np.expand_dims(spec, axis=-1)
            batch_specs.append(spec)

        X_batch = np.asarray(batch_specs, dtype=np.float32)
        batch_probs = model.predict(X_batch, verbose=0).flatten()
        probs.extend(batch_probs)

        print(f"Predicted {min(start + batch_size, len(spectrogram_paths))}/{len(spectrogram_paths)}")

    return np.asarray(probs, dtype=np.float32)


if __name__ == "__main__":
    cnn_model_path = os.path.join(MODEL_DIR, "cnn_model.keras")
    rf_model_path = os.path.join(CLASSICAL_MODEL_DIR, "random_forest_pause_prosody.pkl")
    imputer_path = os.path.join(CLASSICAL_MODEL_DIR, "pause_prosody_imputer.pkl")

    print("Loading models...")
    cnn_model = tf.keras.models.load_model(cnn_model_path)
    rf_model = joblib.load(rf_model_path)
    imputer = joblib.load(imputer_path)

    print("Loading validation metadata...")
    cnn_meta = load_cnn_metadata(VAL_SPECTROGRAM_CSV)
    feat_df = load_feature_validation_data(VAL_FEATURE_CSV)

    merged = cnn_meta.merge(feat_df, on=["file_id", "label_num"], how="inner")
    if merged.empty:
        raise ValueError("No matched samples found between CNN and feature validation data.")

    print("Matched validation samples:", len(merged))

    spectrogram_paths = merged["spectrogram_path"].tolist()
    X_feat = merged[FEATURE_COLUMNS].copy()
    y_true = merged["label_num"].astype(int).values

    X_feat_imputed = imputer.transform(X_feat)

    print("Generating CNN probabilities in batches...")
    cnn_probs = predict_cnn_in_batches(cnn_model, spectrogram_paths, batch_size=BATCH_SIZE)

    print("Generating Random Forest probabilities...")
    rf_probs = rf_model.predict_proba(X_feat_imputed)[:, 1]

    w1 = 0.7
    w2 = 0.3

    fused_probs = (w1 * cnn_probs) + (w2 * rf_probs)
    fused_preds = (fused_probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, fused_preds)
    auc = roc_auc_score(y_true, fused_probs)
    report = classification_report(y_true, fused_preds, digits=4)
    cm = confusion_matrix(y_true, fused_preds)

    print("\nFusion Accuracy:", acc)
    print("Fusion ROC-AUC:", auc)
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    results_df = merged[["file_id", "label_num"]].copy()
    results_df["cnn_prob"] = cnn_probs
    results_df["rf_prob"] = rf_probs
    results_df["fused_prob"] = fused_probs
    results_df["fused_pred"] = fused_preds

    results_csv_path = os.path.join(FUSION_REPORT_DIR, "fusion_validation_predictions.csv")
    report_txt_path = os.path.join(FUSION_REPORT_DIR, "fusion_report.txt")
    config_json_path = os.path.join(FUSION_MODEL_DIR, "fusion_config.json")

    results_df.to_csv(results_csv_path, index=False)

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Fusion Accuracy: {acc:.4f}\n")
        f.write(f"Fusion ROC-AUC: {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    fusion_config = {
        "cnn_weight": w1,
        "rf_weight": w2,
        "decision_threshold": 0.5,
        "batch_size": BATCH_SIZE,
        "cnn_model_path": cnn_model_path,
        "rf_model_path": rf_model_path,
        "imputer_path": imputer_path
    }

    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(fusion_config, f, indent=4)

    print(f"\nSaved prediction CSV to: {results_csv_path}")
    print(f"Saved fusion report to: {report_txt_path}")
    print(f"Saved fusion config to: {config_json_path}")