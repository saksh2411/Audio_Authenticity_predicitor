import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Audio settings
SAMPLE_RATE = 16000
DURATION = 4
TARGET_LENGTH = SAMPLE_RATE * DURATION

# CSV paths
TRAIN_CSV = os.path.join(BASE_DIR, "data", "processed", "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "data", "processed", "val.csv")

TRAIN_PREPROCESSED_CSV = os.path.join(BASE_DIR, "data", "processed", "train_preprocessed.csv")
VAL_PREPROCESSED_CSV = os.path.join(BASE_DIR, "data", "processed", "val_preprocessed.csv")

TRAIN_SPECTROGRAM_CSV = os.path.join(BASE_DIR, "data", "processed", "train_spectrograms.csv")
VAL_SPECTROGRAM_CSV = os.path.join(BASE_DIR, "data", "processed", "val_spectrograms.csv")

# Processed audio folders
PROCESSED_AUDIO_DIR = os.path.join(BASE_DIR, "data", "processed_audio")
TRAIN_OUTPUT_DIR = os.path.join(PROCESSED_AUDIO_DIR, "train")
VAL_OUTPUT_DIR = os.path.join(PROCESSED_AUDIO_DIR, "val")

# Spectrogram folders
SPECTROGRAM_DIR = os.path.join(BASE_DIR, "data", "spectrograms")
TRAIN_SPECTROGRAM_DIR = os.path.join(SPECTROGRAM_DIR, "train")
VAL_SPECTROGRAM_DIR = os.path.join(SPECTROGRAM_DIR, "val")

# Spectrogram parameters
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
FMAX = 8000

# CNN training settings
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Pause/prosody feature paths
FEATURE_DIR = os.path.join(BASE_DIR, "data", "features")
TRAIN_FEATURE_CSV = os.path.join(FEATURE_DIR, "train_pause_prosody_features.csv")
VAL_FEATURE_CSV = os.path.join(FEATURE_DIR, "val_pause_prosody_features.csv")

# Classical ML paths
CLASSICAL_MODEL_DIR = os.path.join(BASE_DIR, "models", "classical")
CLASSICAL_REPORT_DIR = os.path.join(BASE_DIR, "reports", "classical")


FUSION_MODEL_DIR = os.path.join(BASE_DIR, "models", "fusion")
FUSION_REPORT_DIR = os.path.join(BASE_DIR, "reports", "fusion")