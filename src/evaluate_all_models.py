import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from config import (
    VAL_SPECTROGRAM_CSV,
    VAL_FEATURE_CSV,
    MODEL_DIR,
    CLASSICAL_MODEL_DIR,
    FUSION_REPORT_DIR
)

os.makedirs(FUSION_REPORT_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "speech_ratio", "num_pauses", "mean_pause_duration", "std_pause_duration",
    "mean_speech_segment_duration", "std_speech_segment_duration",
    "micropause_rate", "macropause_rate", "mean_f0", "std_f0",
    "voiced_ratio", "rms_mean", "rms_std", "zcr_mean", "zcr_std",
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


def compute_eer(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "eer": compute_eer(y_true, y_prob),
        "tn_fp_fn_tp": confusion_matrix(y_true, y_pred).ravel().tolist()
    }


def save_roc_plot(roc_items, output_path):
    plt.figure(figsize=(8, 6))
    for item in roc_items:
        fpr, tpr, _ = roc_curve(item["y_true"], item["y_prob"])
        plt.plot(fpr, tpr, label=f'{item["name"]} (AUC={item["auc"]:.4f})')
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


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
        raise ValueError("No matched samples found.")

    print("Matched validation samples:", len(merged))

    spectrogram_paths = merged["spectrogram_path"].tolist()
    X_feat = merged[FEATURE_COLUMNS].copy()
    y_true = merged["label_num"].astype(int).values

    X_feat_imputed = imputer.transform(X_feat)

    print("Generating CNN probabilities in batches...")
    cnn_prob = predict_cnn_in_batches(cnn_model, spectrogram_paths, batch_size=BATCH_SIZE)

    print("Generating RF probabilities...")
    rf_prob = rf_model.predict_proba(X_feat_imputed)[:, 1]

    fusion_70_30 = (0.7 * cnn_prob) + (0.3 * rf_prob)
    fusion_50_50 = (0.5 * cnn_prob) + (0.5 * rf_prob)
    fusion_80_20 = (0.8 * cnn_prob) + (0.2 * rf_prob)

    results = {
        "cnn_only": evaluate_predictions(y_true, cnn_prob),
        "rf_only": evaluate_predictions(y_true, rf_prob),
        "fusion_70_30": evaluate_predictions(y_true, fusion_70_30),
        "fusion_50_50": evaluate_predictions(y_true, fusion_50_50),
        "fusion_80_20": evaluate_predictions(y_true, fusion_80_20),
    }

    results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model_name"})
    csv_path = os.path.join(FUSION_REPORT_DIR, "all_model_comparison.csv")
    json_path = os.path.join(FUSION_REPORT_DIR, "all_model_comparison.json")
    roc_path = os.path.join(FUSION_REPORT_DIR, "roc_comparison.png")

    results_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    roc_items = [
        {"name": "CNN Only", "y_true": y_true, "y_prob": cnn_prob, "auc": results["cnn_only"]["roc_auc"]},
        {"name": "RF Only", "y_true": y_true, "y_prob": rf_prob, "auc": results["rf_only"]["roc_auc"]},
        {"name": "Fusion 70/30", "y_true": y_true, "y_prob": fusion_70_30, "auc": results["fusion_70_30"]["roc_auc"]},
        {"name": "Fusion 50/50", "y_true": y_true, "y_prob": fusion_50_50, "auc": results["fusion_50_50"]["roc_auc"]},
        {"name": "Fusion 80/20", "y_true": y_true, "y_prob": fusion_80_20, "auc": results["fusion_80_20"]["roc_auc"]},
    ]

    save_roc_plot(roc_items, roc_path)

    print("Saved comparison CSV to:", csv_path)
    print("Saved comparison JSON to:", json_path)
    print("Saved ROC plot to:", roc_path)
    print(results_df)