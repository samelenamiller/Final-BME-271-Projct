#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:53:07 2023

@author: samanthamiller
"""
import librosa
import librosa.display
from IPython.display import Audio
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
import scipy

file_path= "/Users/samanthamiller/Downloads/BME 271/fullfilter.wav"

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
    return plt.show()

fft_plot(samples, sampr)

def spectrogram(samples, sample_rate, stride_ms = 10.0, 
                          window_ms = 20.0, max_freq = None, eps = 1e-14):

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, 
                                          shape = nshape, strides = nstrides)
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(specgram, x_axis='time', y_axis='linear', sr=sample_rate, hop_length=stride_size)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()
    return specgram
spectrogram(samples, sampr, stride_ms = 10.0, window_ms = 20.0, max_freq = 50000 , eps = 1e-14)


