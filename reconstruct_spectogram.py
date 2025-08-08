import numpy as np
import librosa
import soundfile as sf

if __name__ == "__main__":
    magnitude = np.load("npy_files/spectrogram_magnitude.npy")

    n_fft = 2048
    hop_length = 512

    audio_recon = librosa.griffinlim(magnitude, n_iter=32, hop_length=hop_length, win_length=n_fft)

    sf.write("reconsturction/reconstructed_audio.wav", audio_recon, 22050)

#py -3.12 reconstruct_spectogram.py