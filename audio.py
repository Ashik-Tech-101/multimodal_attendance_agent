"""Lightweight audio activity cues. No transcription or speaker identification."""

import numpy as np


def analyze_audio(file_path, window_seconds=5):
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("Audio analysis needs librosa. Install it with: pip install librosa") from exc

    signal, sample_rate = librosa.load(file_path, sr=16000, mono=True)
    if signal.size == 0:
        return {"windows": [], "mean_rms": 0.0, "active_ratio": 0.0, "note": "Empty audio file."}
    window_size = int(sample_rate * window_seconds)
    windows = []
    for start in range(0, signal.size, window_size):
        chunk = signal[start:start + window_size]
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        windows.append({"time_seconds": round(start / sample_rate), "rms": round(rms, 4)})
    mean_rms = float(np.mean([w["rms"] for w in windows]))
    active_ratio = float(np.mean([w["rms"] > max(0.015, mean_rms * 0.35) for w in windows]))
    return {
        "windows": windows,
        "mean_rms": round(mean_rms, 4),
        "active_ratio": round(active_ratio, 3),
        "note": "Active sound is an approximate energy cue; it does not identify speech or speakers.",
    }
