import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import time

# Decorator to profile execution time
def profile_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # End time
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Function to perform spectral analysis
@profile_time
def analyze_audio_signal(signal, fs):
    """
    Analyzes the audio signal and computes its spectrogram.
    
    Parameters:
    - signal: The audio signal (numpy array).
    - fs: The sampling frequency of the signal.
    
    Returns:
    - f: Frequencies of the spectrogram.
    - t: Time bins of the spectrogram.
    - Sxx: Spectrogram of the signal.
    """
    f, t, Sxx = spectrogram(signal, fs)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity [dB]')
    plt.show()
    return f, t, Sxx

# Example usage
if __name__ == "__main__":
    # Generate a sample audio signal (sine wave)
    fs = 1000  # Sampling frequency
    t = np.linspace(0, 1, fs, endpoint=False)  # Time vector
    frequency = 5  # Frequency of the sine wave
    signal = np.sin(2 * np.pi * frequency * t)  # Sine wave signal

    # Analyze the audio signal
    analyze_audio_signal(signal, fs)
