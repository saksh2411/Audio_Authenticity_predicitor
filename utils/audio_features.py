import numpy as np
import librosa

FRAME_LENGTH = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 16000

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


def extract_pause_prosody_features(audio, sr=SAMPLE_RATE):
    audio_duration = len(audio) / sr

    intervals = librosa.effects.split(audio, top_db=30)
    speech_durations = []
    pause_durations = []

    previous_end = 0
    for start, end in intervals:
        speech_durations.append((end - start) / sr)

        if start > previous_end:
            pause_durations.append((start - previous_end) / sr)

        previous_end = end

    if previous_end < len(audio):
        pause_durations.append((len(audio) - previous_end) / sr)

    total_speech = np.sum(speech_durations) if len(speech_durations) > 0 else 0
    total_pause = np.sum(pause_durations) if len(pause_durations) > 0 else 0

    speech_ratio = total_speech / audio_duration if audio_duration > 0 else 0
    num_pauses = len(pause_durations)
    mean_pause_duration = np.mean(pause_durations) if pause_durations else 0
    std_pause_duration = np.std(pause_durations) if pause_durations else 0
    mean_speech_segment_duration = np.mean(speech_durations) if speech_durations else 0
    std_speech_segment_duration = np.std(speech_durations) if speech_durations else 0

    micropause_rate = np.sum(np.array(pause_durations) < 0.2) / audio_duration if audio_duration > 0 and pause_durations else 0
    macropause_rate = np.sum(np.array(pause_durations) >= 0.2) / audio_duration if audio_duration > 0 and pause_durations else 0

    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )

    voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    mean_f0 = np.mean(voiced_f0) if len(voiced_f0) > 0 else 0
    std_f0 = np.std(voiced_f0) if len(voiced_f0) > 0 else 0
    voiced_ratio = np.mean(voiced_flag) if voiced_flag is not None else 0

    rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    features = {
        "speech_ratio": speech_ratio,
        "num_pauses": num_pauses,
        "mean_pause_duration": mean_pause_duration,
        "std_pause_duration": std_pause_duration,
        "mean_speech_segment_duration": mean_speech_segment_duration,
        "std_speech_segment_duration": std_speech_segment_duration,
        "micropause_rate": micropause_rate,
        "macropause_rate": macropause_rate,
        "mean_f0": mean_f0,
        "std_f0": std_f0,
        "voiced_ratio": voiced_ratio,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "audio_duration": audio_duration
    }

    return features