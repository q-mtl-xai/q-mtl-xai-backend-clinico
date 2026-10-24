
import numpy as np
import joblib
import os

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
    x = np.asarray(signal, dtype=np.float32).flatten()
    n = len(x)
    mu, sigma = x.mean(), x.std() + 1e-8
    var = float(np.var(x))
    hist, _ = np.histogram(x, bins=10)
    p = hist / (hist.sum() + 1e-8)
    entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    fft = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, 1/fs)
    band = (freqs >= 0.5) & (freqs <= 40)
    p_band = np.sum(fft[band]**2)
    p_total = np.sum(fft**2) + 1e-8
    snr = float(10 * np.log10(p_band / (p_total - p_band + 1e-8) + 1e-8))
    p2p = float(x.max() - x.min())
    zc = float(np.sum((x[:-1] - mu) * (x[1:] - mu) < 0) / n)
    energy = float(np.sum(x**2))
    kurt = float(np.mean(((x - mu) / sigma) ** 4))
    skew = float(np.mean(((x - mu) / sigma) ** 3))
    return np.array([var, entropy, snr, p2p, zc, energy, kurt, skew], dtype=np.float32)


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
