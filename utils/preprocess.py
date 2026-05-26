import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MELS = 128
FIXED_TIME_FRAMES = 251


def load_audio(file_path, sr=SAMPLE_RATE):
    audio, sr = librosa.load(file_path, sr=sr, mono=True)
    return audio, sr


def create_mel_spectrogram(audio, sr=SAMPLE_RATE, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def pad_or_truncate_spectrogram(spec, target_frames=FIXED_TIME_FRAMES):
    current_frames = spec.shape[1]

    if current_frames < target_frames:
        pad_width = target_frames - current_frames
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")
    else:
        spec = spec[:, :target_frames]

    return spec


def prepare_cnn_input(file_path):
    audio, sr = load_audio(file_path)
    spec = create_mel_spectrogram(audio, sr)
    spec = pad_or_truncate_spectrogram(spec)
    spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
    spec = np.expand_dims(spec, axis=-1)
    spec = np.expand_dims(spec, axis=0)
    return spec, audio, sr