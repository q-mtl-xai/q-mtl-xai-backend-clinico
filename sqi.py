
import numpy as np
import joblib
import os
from scipy.stats import entropy as scipy_entropy

_BASE = os.path.dirname(__file__)


def _load_sqi(path):
    """Carga el .pkl y devuelve (modelo, scaler_opcional, keys_info)."""
    obj = joblib.load(path)
    info = f"{os.path.basename(path)}: type={type(obj).__name__}"
    if isinstance(obj, dict):
        info += f", keys={list(obj.keys())}"
        model = obj.get("model") or obj.get("classifier") or obj.get("clf")
        scaler = obj.get("scaler") or obj.get("standard_scaler")
        return model, scaler, info
    return obj, None, info


_sqi_ecg_model, _sqi_ecg_scaler, _info_ecg = _load_sqi(
    os.path.join(_BASE, "models", "c3_sqi_ecg.pkl"))
_sqi_ppg_model, _sqi_ppg_scaler, _info_ppg = _load_sqi(
    os.path.join(_BASE, "models", "c3_sqi_ppg.pkl"))

print(f"[SQI] {_info_ecg}")
print(f"[SQI] {_info_ppg}")

SQ_NAMES = {0: "Poor", 1: "Fair", 2: "Good"}


def _format_label(pred):
    if isinstance(pred, str):
        return pred
    if isinstance(pred, (int, np.integer)):
        return SQ_NAMES.get(int(pred), "Fair")
    return str(pred)


def extract_sqi_features(signal, fs=250):
    """
    FIX CRÍTICO: Debe coincidir EXACTAMENTE con notebook 05.
    
    Features (en orden):
        0. var             → varianza
        1. entropy         → scipy_entropy(hist_50bins, log natural)
        2. snr             → percentil-based (p95-p5)
        3. amplitude       → max - min
        4. zero_crossings  → sign-based
        5. energy          → MEDIA (÷ n)
        6. kurtosis        → 4to momento
        7. skewness        → 3er momento
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    # 1. Varianza
    var = np.var(signal)

    # 2. Entropía de Shannon (histograma 50 bins, density=True, log natural)
    hist, _ = np.histogram(signal, bins=50, density=True)
    hist = hist[hist > 0]
    ent = scipy_entropy(hist)

    # 3. SNR percentil-based (amplitud p95-p5 sobre ruido)
    p95 = np.percentile(signal, 95)
    p5 = np.percentile(signal, 5)
    p50 = np.percentile(signal, 50)
    signal_power = (p95 - p5) ** 2
    noise_power = np.var(signal - np.median(signal))
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # 4. Amplitud pico-a-pico
    amp = signal.max() - signal.min()

    # 5. Cruces por cero (sign-based, no media-based)
    zc = np.sum(np.diff(np.sign(signal)) != 0) / n

    # 6. Energía MEDIA (÷ n)
    energy = np.sum(signal ** 2) / n

    # 7. Curtosis
    mean = np.mean(signal)
    std = np.std(signal) + 1e-10
    kurt = np.mean(((signal - mean) / std) ** 4)

    # 8. Skewness
    skew = np.mean(((signal - mean) / std) ** 3)

    return np.array(
        [var, ent, snr, amp, zc, energy, kurt, skew],
        dtype=np.float32,
    )


def _predict(model, scaler, signal, fs):
    if model is None:
        return "Fair"
    feats = extract_sqi_features(signal, fs).reshape(1, -1)
    if scaler is not None:
        feats = scaler.transform(feats)
    pred = model.predict(feats)[0]
    return _format_label(pred)


def predict_sqi_ecg(signal, fs=250):
    return _predict(_sqi_ecg_model, _sqi_ecg_scaler, signal, fs)


def predict_sqi_ppg(signal, fs=250):
    return _predict(_sqi_ppg_model, _sqi_ppg_scaler, signal, fs)
