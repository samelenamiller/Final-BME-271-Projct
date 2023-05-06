#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 17:55:58 2023

@author: samanthamiller
"""

import librosa
import numpy as np
import scipy
from scipy.io import wavfile
import pandas as pd

file_path = "/Users/samanthamiller/Downloads/Madonna - Hung Up (Album Version).wav"

y, sr = librosa.load(file_path, sr=None, mono=True, offset=0.0, duration=None)
fft = np.fft.fft(y)
freqs = np.fft.fftfreq(len(fft), d=1/sr)
df = pd.DataFrame({"frequency": freqs, "amplitude": np.abs(fft)})
df.loc[df['frequency'] > 1000] *= .9

amps = df["amplitude"].values
ifft = np.fft.ifft(amps)
audio_filtered = np.real(ifft)
wavfile.write('soundscale.wav', sr, audio_filtered.astype(np.int16))

