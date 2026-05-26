import math
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from config import (
    TRAIN_SPECTROGRAM_CSV,
    VAL_SPECTROGRAM_CSV,
    MODEL_DIR,
    REPORT_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


class SpectrogramDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_path, batch_size=8, shuffle=True):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["spectrogram_status"] == "ok"].copy().reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        X_batch = []
        y_batch = []

        for _, row in batch_df.iterrows():
            spec_path = row["spectrogram_path"]
            label = row["label_num"]

            spec = np.load(spec_path).astype(np.float32)

            spec_mean = np.mean(spec, dtype=np.float32)
            spec_std = np.std(spec, dtype=np.float32) + 1e-8
            spec = (spec - spec_mean) / spec_std

            spec = np.expand_dims(spec, axis=-1)

            X_batch.append(spec)
            y_batch.append(label)

        X_batch = np.stack(X_batch, axis=0).astype(np.float32)
        y_batch = np.array(y_batch, dtype=np.float32)

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_all_labels(self):
        return self.df["label_num"].values.astype(np.int32)


def get_input_shape(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["spectrogram_status"] == "ok"].copy()

    sample_path = df.iloc[0]["spectrogram_path"]
    spec = np.load(sample_path).astype(np.float32)
    spec = np.expand_dims(spec, axis=-1)

    return spec.shape


def build_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def plot_history(history, output_path):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close("all")


if __name__ == "__main__":
    train_gen = SpectrogramDataGenerator(TRAIN_SPECTROGRAM_CSV, batch_size=BATCH_SIZE, shuffle=True)
    val_gen = SpectrogramDataGenerator(VAL_SPECTROGRAM_CSV, batch_size=BATCH_SIZE, shuffle=False)

    input_shape = get_input_shape(TRAIN_SPECTROGRAM_CSV)
    print("Input shape:", input_shape)
    print("Train batches:", len(train_gen))
    print("Val batches:", len(val_gen))

    model = build_cnn_model(input_shape)
    model.summary()

    model_path = os.path.join(MODEL_DIR, "cnn_model.keras")
    history_plot_path = os.path.join(REPORT_DIR, "cnn_training_curve.png")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    plot_history(history, history_plot_path)

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print("\nValidation Loss:", val_loss)
    print("Validation Accuracy:", val_acc)

    y_true = val_gen.get_all_labels()
    y_pred_prob = model.predict(val_gen)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print(f"\nBest model saved to: {model_path}")
    print(f"Training curve saved to: {history_plot_path}")