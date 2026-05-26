import os
import warnings
import json
import numpy as np
import librosa
import joblib
import tensorflow as tf

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
CLASSICAL_DIR = os.path.join(MODEL_DIR, "classical")
FUSION_DIR = os.path.join(MODEL_DIR, "fusion")

CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.keras")
RF_MODEL_PATH = os.path.join(CLASSICAL_DIR, "random_forest_pause_prosody.pkl")
IMPUTER_PATH = os.path.join(CLASSICAL_DIR, "pause_prosody_imputer.pkl")
FUSION_CONFIG_PATH = os.path.join(FUSION_DIR, "fusion_config.json")

TARGET_SR = 16000
DURATION_SEC = 4
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

REAL_THRESHOLD = 0.48
FAKE_THRESHOLD = 0.52

DEFAULT_CNN_WEIGHT = 0.5
DEFAULT_RF_WEIGHT = 0.5

CNN_OUTPUT_IS_REAL_PROB = False


def load_required_assets():
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(f"CNN model not found: {CNN_MODEL_PATH}")

    if not os.path.exists(RF_MODEL_PATH):
        raise FileNotFoundError(f"RF model not found: {RF_MODEL_PATH}")

    if not os.path.exists(IMPUTER_PATH):
        raise FileNotFoundError(f"Imputer not found: {IMPUTER_PATH}")

    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)
    rf_model = joblib.load(RF_MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    fusion_config = {}
    if os.path.exists(FUSION_CONFIG_PATH):
        try:
            with open(FUSION_CONFIG_PATH, "r", encoding="utf-8") as f:
                fusion_config = json.load(f)
        except Exception:
            fusion_config = {}

    return cnn_model, rf_model, imputer, fusion_config


cnn_model, rf_model, imputer, fusion_config = load_required_assets()

CNN_WEIGHT = float(fusion_config.get("cnn_weight", DEFAULT_CNN_WEIGHT))
RF_WEIGHT = float(fusion_config.get("rf_weight", DEFAULT_RF_WEIGHT))


def load_audio_for_model(file_path, sr=TARGET_SR, duration_sec=DURATION_SEC):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    y = librosa.util.normalize(y)

    target_len = sr * duration_sec
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    return y.astype(np.float32), sr


def extract_cnn_input(audio, sr=TARGET_SR):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_min = mel_db.min()
    mel_max = mel_db.max()

    if mel_max - mel_min > 1e-8:
        mel_db = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_db = np.zeros_like(mel_db)

    mel_db = np.expand_dims(mel_db, axis=-1)
    mel_db = np.expand_dims(mel_db, axis=0)
    return mel_db.astype(np.float32)


def extract_handcrafted_features(audio, sr=TARGET_SR):
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=HOP_LENGTH)[0]

    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7")
        )
        valid_f0 = f0[~np.isnan(f0)]
        pitch_mean = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0
        pitch_std = float(np.std(valid_f0)) if len(valid_f0) > 0 else 0.0
        voiced_ratio = float(np.mean(voiced_flag.astype(np.float32))) if voiced_flag is not None else 0.0
    except Exception:
        pitch_mean = 0.0
        pitch_std = 0.0
        voiced_ratio = 0.0

    abs_audio = np.abs(audio)
    silence_mask = abs_audio < 0.01
    silence_ratio = float(np.mean(silence_mask))

    changes = np.diff(silence_mask.astype(np.int32))
    silence_starts = np.where(changes == 1)[0]
    silence_ends = np.where(changes == -1)[0]

    if silence_mask[0]:
        silence_starts = np.insert(silence_starts, 0, 0)
    if silence_mask[-1]:
        silence_ends = np.append(silence_ends, len(silence_mask) - 1)

    pause_lengths = []
    for s, e in zip(silence_starts, silence_ends):
        pause_len = (e - s) / sr
        if pause_len > 0.03:
            pause_lengths.append(pause_len)

    pause_count = len(pause_lengths)
    mean_pause = float(np.mean(pause_lengths)) if pause_lengths else 0.0
    std_pause = float(np.std(pause_lengths)) if pause_lengths else 0.0
    max_pause = float(np.max(pause_lengths)) if pause_lengths else 0.0
    long_pause_count = int(sum(p > 0.3 for p in pause_lengths))

    duration = float(len(audio) / sr)
    speech_ratio = float(1.0 - silence_ratio)
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    features = {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "voiced_ratio": voiced_ratio,
        "silence_ratio": silence_ratio,
        "pause_count": float(pause_count),
        "mean_pause_duration": mean_pause,
        "std_pause_duration": std_pause,
        "max_pause_duration": max_pause,
        "long_pause_count": float(long_pause_count),
        "speech_ratio": speech_ratio,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "zcr_mean": float(np.mean(zcr)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff))
    }

    feature_vector = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)
    return features, feature_vector


def predict_cnn(audio, sr=TARGET_SR):
    cnn_input = extract_cnn_input(audio, sr)
    pred = cnn_model.predict(cnn_input, verbose=0)
    pred = np.array(pred).squeeze()

    if np.ndim(pred) == 0:
        prob_fake = float(pred)
    elif np.shape(pred) == (2,):
        prob_fake = float(pred[1])
    else:
        prob_fake = float(np.ravel(pred)[-1])

    prob_fake = float(np.clip(prob_fake, 0.0, 1.0))

    if CNN_OUTPUT_IS_REAL_PROB:
        prob_fake = 1.0 - prob_fake

    return prob_fake


def predict_rf(feature_vector):
    x = imputer.transform(feature_vector)
    probs = rf_model.predict_proba(x)[0]

    if hasattr(rf_model, "classes_"):
        classes = list(rf_model.classes_)
        classes_lower = [str(c).strip().lower() for c in classes]

        if "fake" in classes_lower:
            return float(probs[classes_lower.index("fake")])
        if "spoof" in classes_lower:
            return float(probs[classes_lower.index("spoof")])
        if 1 in classes:
            return float(probs[classes.index(1)])

    return float(probs[-1])


def fuse_scores(cnn_prob, rf_prob):
    fused = (CNN_WEIGHT * cnn_prob) + (RF_WEIGHT * rf_prob)
    return float(np.clip(fused, 0.0, 1.0))


def get_final_label(fused_prob):
    if fused_prob >= FAKE_THRESHOLD:
        return "Fake"
    elif fused_prob <= REAL_THRESHOLD:
        return "Real"
    return "Suspicious"


def predict_file(file_path):
    audio, sr = load_audio_for_model(file_path, sr=TARGET_SR, duration_sec=DURATION_SEC)
    features_dict, feature_vector = extract_handcrafted_features(audio, sr)

    cnn_prob = predict_cnn(audio, sr)
    rf_prob = predict_rf(feature_vector)
    fused_prob = fuse_scores(cnn_prob, rf_prob)
    final_label = get_final_label(fused_prob)

    result = {
        "cnn_prob": float(cnn_prob),
        "rf_prob": float(rf_prob),
        "fused_prob": float(fused_prob),
        "final_label": final_label,
        "features": features_dict,
        "debug": {
            "cnn_model_path": CNN_MODEL_PATH,
            "rf_model_path": RF_MODEL_PATH,
            "imputer_path": IMPUTER_PATH,
            "fusion_config_path": FUSION_CONFIG_PATH,
            "rf_classes": list(rf_model.classes_) if hasattr(rf_model, "classes_") else None,
            "feature_count": int(feature_vector.shape[1]),
            "cnn_weight": CNN_WEIGHT,
            "rf_weight": RF_WEIGHT
        }
    }

    return result, audio, sr