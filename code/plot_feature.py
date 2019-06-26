"""
Extract and plot the mel-spectrogram from two audio files belonging to different classes.
"""

import matplotlib.pyplot as plt
try:
    from code.feature_extraction import *
except ImportError:
    from feature_extraction import *   # when running from terminal, the directory may not be identified as a package
import os


# set paths to a audio files
audio_path_song1= '../data/train/1.mp3'
audio_path_song2 = '../data/test/902.mp3'

parameters = load_parameters() # load parameters

# extract features
feature_song1 = extract_feature(audio_path_song1, parameters)
feature_song2 = extract_feature(audio_path_song2, parameters)

# plot
plt.subplot(211)
plt.imshow(feature_song1, aspect='auto', origin='lower')
plt.title(os.path.basename(audio_path_song1))
plt.colorbar()
plt.subplot(212)
plt.imshow(feature_song2, aspect='auto', origin='lower')
plt.title(os.path.basename(audio_path_song2))
plt.colorbar()
plt.show()