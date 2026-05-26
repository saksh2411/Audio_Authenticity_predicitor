import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from config import (
    TRAIN_FEATURE_CSV,
    VAL_FEATURE_CSV,
    CLASSICAL_MODEL_DIR,
    CLASSICAL_REPORT_DIR
)

os.makedirs(CLASSICAL_MODEL_DIR, exist_ok=True)
os.makedirs(CLASSICAL_REPORT_DIR, exist_ok=True)


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


def load_feature_data(csv_path):
    df = pd.read_csv(csv_path)

    df = df[df["feature_status"] == "ok"].copy()

    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = df[FEATURE_COLUMNS].copy()
    y = df["label_num"].astype(int).values

    return X, y, df


def save_feature_importance_plot(importances, feature_names, output_path):
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features[::-1], sorted_importances[::-1])
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    print("Loading training feature data...")
    X_train, y_train, train_df = load_feature_data(TRAIN_FEATURE_CSV)

    print("Loading validation feature data...")
    X_val, y_val, val_df = load_feature_data(VAL_FEATURE_CSV)

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )

    print("\nTraining Random Forest model...")
    model.fit(X_train_imputed, y_train)

    y_pred = model.predict(X_val_imputed)

    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, digits=4)
    cm = confusion_matrix(y_val, y_pred)

    print("\nValidation Accuracy:", acc)
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    model_path = os.path.join(CLASSICAL_MODEL_DIR, "random_forest_pause_prosody.pkl")
    imputer_path = os.path.join(CLASSICAL_MODEL_DIR, "pause_prosody_imputer.pkl")
    report_path = os.path.join(CLASSICAL_REPORT_DIR, "random_forest_report.txt")
    importance_plot_path = os.path.join(CLASSICAL_REPORT_DIR, "random_forest_feature_importance.png")

    joblib.dump(model, model_path)
    joblib.dump(imputer, imputer_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Validation Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    save_feature_importance_plot(
        model.feature_importances_,
        FEATURE_COLUMNS,
        importance_plot_path
    )

    print(f"\nSaved model to: {model_path}")
    print(f"Saved imputer to: {imputer_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved feature importance plot to: {importance_plot_path}")