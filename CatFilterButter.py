#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:20:19 2023

@author: samanthamiller
"""

import librosa
import librosa.display
from IPython.display import Audio
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import butter, filtfilt, freqz
from scipy.io.wavfile import write

file_path= "/Users/samanthamiller/Downloads/Madonna - Hung Up (Album Version).wav"

samples, sampr = librosa.load(file_path, sr=None,
                              mono=True, offset= 0.0, duration = None)
print(len(samples), sampr)
#Audio(file_path)

plt.figure()
librosa.display.waveshow(y=samples, sr =sampr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

def fft_plot(audio, sampling_rate):
    n= len(audio)
    T = 1/sampling_rate
    yf= scipy.fft.fft(audio)
    xf = np.linspace(0.0, 1/(2*T), int(n/2))
    fig, ax = plt.subplots()
    ax.plot(xf, 2/n *np.abs(yf[:n//2]))
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()
    
    fs = sampling_rate  # Sampling frequency (Hz)
    f1 = 500  # Lower cutoff frequency (Hz)
    f2 = 4500  # Upper cutoff frequency (Hz)
    o = 4  # Filter order (number of poles)

    # Calculate filter coefficients
    b, a = butter(o, [2*f1/fs, 2*f2/fs], btype='bandpass')
    yf_butter = filtfilt(b, a, samples)
    write("filtered_audio_butter.wav", fs, yf_butter.astype(np.float32))
    n_butter = len(yf_butter)
    xf_butter = np.linspace(0.0, 1/(2*T), int(n_butter/2))
    fig, ax = plt.subplots()
    ax.plot(xf_butter, 2/n_butter *np.abs(yf_butter[:n_butter//2]))
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

fft_plot(samples, sampr)