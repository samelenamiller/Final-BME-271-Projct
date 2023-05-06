#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:23:03 2023

@author: samanthamiller
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io.wavfile import write
from scipy.io import wavfile
from pydub import AudioSegment
import librosa
import librosa.display
from scipy.signal import bessel, filtfilt, freqz
from pydub import AudioSegment
from scipy.signal import bessel, filtfilt, freqz

fs_madonna, madonna_audio = wavfile.read('/Users/samanthamiller/Downloads/Madonna - Hung Up (Album Version).wav')
fs_300swift, swift_audio = wavfile.read('/Users/samanthamiller/Downloads/300 Swift Apartments 2.wav')
file_path= "/Users/samanthamiller/Downloads/Madonna - Hung Up (Album Version).wav"
#'/Users/samanthamiller/Downloads/Madonna - Hung Up (Album Version).wav', format="wav")

samples, sampr = librosa.load(file_path, sr=None,
                              mono=True, offset= 0.0, duration = None)
#addpurrs

audio1 = "/Users/samanthamiller/Downloads/catpurr.wav"
sound2 = AudioSegment.from_wav(audio1)
sound1 = AudioSegment.from_wav(file_path)

softer = sound2- 10

overlay = sound1.overlay(softer, position=20)

overlay.export("overlaypurr.wav", format="wav")


#concolve purrs

song = "/Users/samanthamiller/Downloads/BME 271/overlaypurr.wav"
srsong, songaudio = wavfile.read(song)
songaudio = songaudio / np.max(np.abs(songaudio))
swift_audio = swift_audio / np.max(np.abs(swift_audio))

convolved_audio = signal.convolve(songaudio, swift_audio, mode='full')

convolveda = convolved_audio / np.max(abs(convolved_audio))

write('conpurr.wav', srsong, convolveda.astype(np.float32))


#bessel filter
def fft_plot(audio, sampling_rate):
    n= len(audio)
    T = 1/sampling_rate
    yf= audio
    xf = np.linspace(0.0, 1/(2*T), int(n/2))
    fs = sampling_rate  
    o = 1# filter order 
    fc = 250
    b, a = bessel(o, 2*np.pi*fc, 'lowpass', analog=False, fs=fs)
    yfbes = filtfilt(b, a,audio)
    nbes= len(yfbes)
    xfbes = np.linspace(0.0, 1/(2*T), int(nbes/2))
    fig, ax = plt.subplots()
    ax.plot(xfbes, 2/nbes *np.abs(yfbes[:nbes//2]))
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()
    write("fullfilter.wav", fs, yfbes.astype(np.float32))
    
audiop = "/Users/samanthamiller/Downloads/BME 271/conpurr.wav"
samples, sampr = librosa.load(audiop, sr=None,
                            mono=True, offset= 0.0, duration = None)
print(len(samples), sampr)

fft_plot(samples, srsong)

