try:
    from code.feature_extraction import *
except ImportError:
    from feature_extraction import *   # when running from terminal, the directory may not be identified as a package
import matplotlib.pyplot as plt
import h5py
import os
import numpy as np
import time


# load parameters (stored in yaml file)
parameters = load_parameters()

audio_path = '../data/train/2.mp3'
feature = extract_feature(audio_path, parameters)
#x2 = int(np.round(get_frame_for_label_index(2, parameters)))
#x1 = x2 - int(np.round(0.5*parameters['fs']/parameters['hop_size']))

print(feature)


#X, y = data[image_padding_amount:, :-image_padding_amount], data[:, -image_padding_amount]
image_padding_amount = math.floor(parameters['snippet_length_3sec'] * parameters['fs']/parameters['window_size'])


#feature_selection = feature[:,x1:x2]
feature_selection = feature[:, image_padding_amount:-image_padding_amount]


# plot
plt.subplot(211)
plt.imshow(feature, aspect=2, origin='lower')
plt.title(os.path.basename(audio_path))
plt.colorbar()
plt.subplot(212)
plt.imshow(feature_selection, aspect=2, origin='lower')
plt.title(os.path.basename(audio_path))
plt.colorbar()
plt.show()