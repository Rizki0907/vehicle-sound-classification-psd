import librosa
import scipy.signal as sig
from scipy.fft import fft, fftfreq
import numpy as np

FILTER_CONFIG = {
    'bus': {'low': 40, 'high': 8000},
    'Train': {'low': 100, 'high': 8000},
    'Truck': {'low': 40, 'high': 7500},
    'Cars': {'low': 80, 'high': 8000},
    'Motocycles': {'low': 120, 'high': 10000},
    'Bics': {'low': 10, 'high': 3000},
    'Airplane': {'low': 100, 'high': 10000},
    'Helicopter': {'low': 20, 'high': 9000},
    'default' : {'low': 10, 'high': 10000}
}

def get_filter_taps(sample_label, fs, num_taps=101):

    key_found = 'default'
    if isinstance(sample_label, str):
        label_lower = sample_label.lower()
        for key in FILTER_CONFIG:
            if key.lower() in label_lower:
                key_found = key
                break

    config = FILTER_CONFIG[key_found]

    low_freq = config['low']
    high_freq = config['high']

    nyquist = 0.5 * fs
    norm_low = low_freq / nyquist
    norm_high = high_freq / nyquist

    if norm_high >= 1.0: norm_high = 0.99

    taps = sig.firwin(num_taps, [norm_low, norm_high], pass_zero=False)

    return taps, low_freq, high_freq

def apply_filter_function(signal, taps):
    return sig.lfilter(taps, 1.0, signal)

def apply_fft_bandpass_filter(signal, sr, low_freq, high_freq):
    N = len(signal)
    T = 1.0 / sr

    yf = fft(signal)
    xf = fftfreq(N, T)

    yf_filtered = np.copy(yf)

    mask = (np.abs(xf) >= low_freq) & (np.abs(xf) <= high_freq)
    yf_filtered[~mask] = 0

    filtered_signal = np.real(np.fft.ifft(yf_filtered))

    return filtered_signal

def extract_advanced_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    features = np.hstack([mfcc_mean, centroid_mean, rolloff_mean, zcr_mean])

    return features

def process_file_advanced(file_path, label_hint='default', apply_filter=True, apply_fft_filter=False):
    try:
        y, sr = librosa.load(file_path, sr=22050)

        if apply_filter:
            filter_taps, low_freq, high_freq = get_filter_taps(label_hint, sr)
            y_clean = apply_filter_function(y, filter_taps)
            if apply_fft_filter:
                y_clean = apply_fft_bandpass_filter(y_clean, sr, low_freq, high_freq)
        else:
            y_clean = y
            if apply_fft_filter:
                filter_taps_dummy, low_freq, high_freq = get_filter_taps(label_hint, sr)
                y_clean = apply_fft_bandpass_filter(y_clean, sr, low_freq, high_freq)

        feature_vector = extract_advanced_features(y_clean, sr)

        return feature_vector

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


