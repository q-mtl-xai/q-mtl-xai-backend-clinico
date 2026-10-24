
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy.interpolate import interp1d


def filter_ecg(signal, fs, lowcut=0.5, highcut=40.0, notch_freq=50.0):
    nyq = 0.5 * fs
    if highcut >= nyq:
        highcut = nyq - 1
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    filtered = filtfilt(b, a, signal)
    if notch_freq < nyq:
        bn, an = iirnotch(notch_freq, 30, fs)
        filtered = filtfilt(bn, an, filtered)
    return filtered


def resample_signal(signal, fs_orig, fs_target):
    if fs_orig == fs_target:
        return signal
    n_orig = len(signal)
    n_target = int(n_orig * fs_target / fs_orig)
    t_orig = np.linspace(0, 1, n_orig)
    t_target = np.linspace(0, 1, n_target)
    f = interp1d(t_orig, signal, kind='cubic', fill_value='extrapolate')
    return f(t_target)


def normalize_minmax(signal):
    smin, smax = signal.min(), signal.max()
    if smax - smin < 1e-8:
        return signal - smin
    return (signal - smin) / (smax - smin)


def preprocess_ecg(signal, fs_orig, fs_target=250, window_size=250):
    """Preprocesa señal 1D o 2D con fix de isquemia (already_normalized)."""
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim == 1:
        sig = filter_ecg(signal, fs_orig)
        sig = resample_signal(sig, fs_orig, fs_target)
        sig = normalize_minmax(sig)
        if len(sig) > window_size:
            sig = sig[:window_size]
        elif len(sig) < window_size:
            sig = np.pad(sig, (0, window_size - len(sig)))
        return sig

    n_leads, n_samples = signal.shape
    already_normalized = (signal.min() >= -0.01 and signal.max() <= 1.01)

    processed = []
    for i in range(n_leads):
        s = signal[i]
        if not already_normalized:
            s = filter_ecg(s, fs_orig)
            s = resample_signal(s, fs_orig, fs_target)
            s = normalize_minmax(s)
        else:
            if fs_orig != fs_target:
                s = resample_signal(s, fs_orig, fs_target)
        if len(s) > window_size:
            s = s[:window_size]
        elif len(s) < window_size:
            s = np.pad(s, (0, window_size - len(s)))
        processed.append(s)

    return np.stack(processed, axis=0)
