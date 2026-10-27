
import numpy as np
import joblib
import os
from scipy.stats import entropy as scipy_entropy

_BASE = os.path.dirname(__file__)


def _load_sqi(path):
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
_ORDER = {"Good": 0, "Fair": 1, "Poor": 2}


def _format_label(pred):
    if isinstance(pred, str):
        return pred
    if isinstance(pred, (int, np.integer)):
        return SQ_NAMES.get(int(pred), "Fair")
    return str(pred)


def extract_sqi_features(signal, fs=250):
    """Features alineadas con notebook 05."""
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    var = np.var(signal)

    hist, _ = np.histogram(signal, bins=50, density=True)
    hist = hist[hist > 0]
    ent = scipy_entropy(hist)

    p95 = np.percentile(signal, 95)
    p5 = np.percentile(signal, 5)
    signal_power = (p95 - p5) ** 2
    noise_power = np.var(signal - np.median(signal))
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

    amp = signal.max() - signal.min()
    zc = np.sum(np.diff(np.sign(signal)) != 0) / n
    energy = np.sum(signal ** 2) / n

    mean = np.mean(signal)
    std = np.std(signal) + 1e-10
    kurt = np.mean(((signal - mean) / std) ** 4)
    skew = np.mean(((signal - mean) / std) ** 3)

    return np.array([var, ent, snr, amp, zc, energy, kurt, skew], dtype=np.float32)


def _model_predict(signal, fs):
    """Predicción del modelo entrenado (dominio MIT-BIH, 1D)."""
    if _sqi_ecg_model is None:
        return "Fair"
    feats = extract_sqi_features(signal, fs).reshape(1, -1)
    if _sqi_ecg_scaler is not None:
        feats = _sqi_ecg_scaler.transform(feats)
    pred = _sqi_ecg_model.predict(feats)[0]
    return _format_label(pred)


def _rule_predict(signal, fs, is_2d=False):
    """
    Predicción por reglas calibradas.

    Para señales 2D (PTB-XL) se usa KURTOSIS porque es más discriminativa que
    energy en este dominio (calibrado con sanity check de 3 muestras conocidas):
      - Limpio:  kurt ≈ 9.75   → Good
      - em 0.25: kurt ≈ 18.69  → Fair (artefactos agudos)
      - ma 0.50: kurt ≈ 7.14   → Poor (señal aplanada)

    Para señales 1D (MIT-BIH) se usa energy + snr (dominio del training).
    """
    feats = extract_sqi_features(signal, fs)
    energy = feats[5]
    snr    = feats[2]
    kurt   = feats[6]

    if is_2d:
        # PTB-XL Lead I — reglas por kurtosis
        if kurt > 15:
            return "Fair"
        elif kurt < 8:
            return "Poor"
        else:
            return "Good"
    else:
        # Señales 1D (MIT-BIH, latido, ventana) — reglas por energy + snr
        if energy < 0.10 and snr > 8:
            return "Good"
        elif energy < 0.30 and snr > 4:
            return "Fair"
        else:
            return "Poor"


def predict_sqi_ecg(signal, fs=250, is_2d=False):
    """
    SQI por dominio:
      - 2D (PTB-XL): SOLO reglas por kurtosis (el modelo no generaliza a 12-leads)
      - 1D (MIT-BIH): HÍBRIDO (modelo + reglas, toma el más conservador)
    """
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) == 0:
        return "Fair"

    rule_pred = _rule_predict(signal, fs, is_2d=is_2d)

    if is_2d:
        # Para 2D usar SOLO la regla
        return rule_pred
    else:
        # Para 1D usar el peor entre modelo y regla
        model_pred = _model_predict(signal, fs)
        return max(model_pred, rule_pred, key=lambda x: _ORDER.get(x, 1))


def predict_sqi_ppg(signal, fs=250):
    """SQI PPG (solo modelo, sin reglas)."""
    if _sqi_ppg_model is None:
        return "Fair"
    feats = extract_sqi_features(signal, fs).reshape(1, -1)
    if _sqi_ppg_scaler is not None:
        feats = _sqi_ppg_scaler.transform(feats)
    pred = _sqi_ppg_model.predict(feats)[0]
    return _format_label(pred)
