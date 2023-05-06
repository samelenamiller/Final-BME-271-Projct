#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 12:50:26 2023

@author: samanthamiller
"""

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import scipy

# Read in the audio files
fs_madonna, madonna_audio = wavfile.read('/Users/samanthamiller/Downloads/Madonna - Hung Up (Album Version).wav')
fs_300swift, swift_audio = wavfile.read('/Users/samanthamiller/Downloads/300 Swift Apartments 2.wav')
# Normalize the audio signals

# Normalize the audio signals
madonna_audio = madonna_audio / np.max(np.abs(madonna_audio))
swift_audio = swift_audio / np.max(np.abs(swift_audio))

# Convolve the audio signals
convolved_audio = signal.convolve(madonna_audio, swift_audio, mode='full')

#Scale the convolved audio to prevent clipping
convolved_audio = convolved_audio / np.max(abs(convolved_audio))

# Save the convolved audio as a new WAV file
wavfile.write('convolved_audio.wav', fs_madonna, convolved_audio.astype(np.float32))

#samples, sampr = librosa.load('/Users/samanthamiller/Downloads/convolved_audio.wav, sr=None,
                            #  mono=True, offset= 0.0, duration = None)
                            
