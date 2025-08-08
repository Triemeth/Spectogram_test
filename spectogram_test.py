import soundfile as sf
from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def plt_spectrogram(signal, sample_rate, output_path: Path):
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    
    stft = librosa.stft(signal, n_fft=1024, hop_length=512)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, y_axis='log', x_axis='time',
                             sr=sample_rate, cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.show()
    plt.close()

def main():
    audio_path = Path("data") / "Dont Stop The Music.aif"
    output_path = Path("imgs") / "Dont_Stop_The_Music_Spectrogram.png"
    
    signal, sample_rate = sf.read(audio_path)
    print(f"Sample rate: {sample_rate}, Signal shape: {signal.shape}")
    
    max_len = sample_rate * 10
    if len(signal) > max_len:
        signal = signal[:max_len]
    
    plt_spectrogram(signal, sample_rate, output_path)

if __name__ == "__main__":
    main()

#py -3.12 spectogram_test.py