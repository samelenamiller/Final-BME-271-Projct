#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:37:23 2023

@author: samanthamiller
"""

from pydub import AudioSegment

sound1 = AudioSegment.from_file('/Users/samanthamiller/Downloads/Madonna - Hung Up (Album Version).wav', format="wav")
sound2 = AudioSegment.from_file('/Users/samanthamiller/Downloads/catpurr.wav', format="wav")

softer = sound2- 10

overlay = sound1.overlay(softer, position=20)
file_handle = overlay.export("catpurrhungup.wav", format="wav")

