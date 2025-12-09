import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    return fig

def plot_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(S))
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title("Spectrogram")
    fig.colorbar(img, ax=ax)
    return fig
