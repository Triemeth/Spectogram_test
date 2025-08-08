import librosa
import numpy as np

if __name__ == "__main__":
    y, sr = librosa.load("data/Dont Stop The Music.aif", sr=None, mono=True)

    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)

    np.save("npy_files/spectrogram_magnitude.npy", magnitude)

#py -3.12 save_magnitude.py