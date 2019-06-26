"""
STEP 1: Extract features from all audio files and store them in a single hdf5 file (~300MB).
"""

try:
    from code.feature_extraction import *
except ImportError:
    from feature_extraction import *   # when running from terminal, the directory may not be identified as a package
import h5py
import os
import numpy as np
import time
import pandas as pd

# load parameters (stored in yaml file)
parameters = load_parameters()

# get a list of all train and test files

song_num=[]

df = pd.read_csv("arousal_cont_average.csv")

# print('DF VALUES...')

song_num=df['song_id'].values

# print(df['song_id'].iloc[1])

# print(song_num)

# print(df.iloc[0].values)

print('gathering files...')
song_path = '../data/{test_or_train}/{song_num}.mp3'

# print(len(df.index))

train_files = {}
for k, song_num_i in enumerate(song_num):
    train_file = song_path.format(test_or_train='train', song_num=song_num_i)
    if os.path.isfile(train_file):
        train_files[k] = train_file


test_files = {}
for j, song_num_i in enumerate(song_num):
    test_file = song_path.format(test_or_train='test', song_num=song_num_i)
    if os.path.isfile(test_file):
        test_files[j-len(train_files)] = test_file

print(' --> ', len(train_files), 'train files and', len(test_files), 'test files found.')


labels_per_audio = 61

label_array = []

## drop the first column first

df_drop_first_column = df.drop('song_id', 1)


for n in range(len(song_num)):
    label_array.append(df_drop_first_column.iloc[n].values)


first_label = (3 + 15)*parameters['fs']/parameters['hop_size']

label_gap = int(np.round(0.5*parameters['fs']/parameters['hop_size']))

# print(label_array)

# print(label_array[2])

inds = list(range(len(label_array)))

# print(range(len(label_array)))
# print(len(label_array))
# print(inds)

#for i, x in enumerate(label_array)
    


# hdf5-setup
print('setting up hdf5 file...')

# pre-compute number of instances (for audio spectograms)
num_train_instances = len(train_files)
num_test_instances = len(test_files)


# pre-compute the number of labels
num_train_labels = len(train_files)*labels_per_audio
num_test_labels = len(test_files)*labels_per_audio


# pre-compute "image" height and width
image_height = parameters['num_mel']
# time to frames
image_width = int(np.round(parameters['snippet_length_sec'] * parameters['fs'] / float(parameters['hop_size']))) + 2*math.floor(parameters['snippet_length_3sec'] * parameters['fs']/parameters['hop_size'])
# image width per half second in terms of samples
image_width_per_half_sec = int(np.round(0.5*parameters['fs']/parameters['hop_size']))


# init hdf5 file
dataset = h5py.File('../data/dataset.hdf5', mode='w')  # this is where we will store it



#LABELS - the way Luis suggested?
#SAME SHAPE as FEATURES  or just image width?


# we store 61 float labels per instance: 
dataset.create_dataset('train_labels',
                       shape=(num_train_instances, 61, 1),
                       dtype=np.float64)
dataset.create_dataset('test_labels',
                       shape=(num_test_instances, 61, 1),
                       dtype=np.float64)




# we store one image per instance of size image_height x image_width holding floating point numbers
dataset.create_dataset('train_features',
                       shape=(num_train_instances, image_height, image_width),
                       dtype=np.float)
dataset.create_dataset('test_features',
                       shape=(num_test_instances, image_height, image_width),
                       dtype=np.float)



#from column 1 onwards (i.e. 15500 ms onwards)
#15500ms x1 frame index
#15500ms - 500ms x0 frame index
#mel[x1:x0]

print(test_files)


# TEST SET FOR THE AUDIOS IN THE CSV
for k in test_files:
    print('  ', k + 1, '/', len(test_files), test_files[k])
    t_s = time.time()
    # extract the feature
    feature = extract_feature(test_files[k], parameters)

    dataset['test_features'][k, ...] = feature
    print('working on test labels...')
    for j in range(len(label_array[k])):
        dataset['test_labels'][k, j, ...] = label_array[len(train_files) + k][j]
        print(k, '-', j, '-', label_array[len(train_files)+k][j])

    print('--> time elapsed: ', time.time() - t_s)



# TRAIN SET FOR THE AUDIOS IN THE CSV
print('working on train features...')
for k in train_files:
    print('  ', k + 1, '/', len(train_files), train_files[k])
    t_s = time.time()
    # extract the feature
    feature = extract_feature(train_files[k], parameters)

    dataset['train_features'][k, ...] = feature
    print('working on train labels...')
    for j in range(len(label_array[k])):
        dataset['train_labels'][k, j, ...] = label_array[k][j]
        print(k, '-', j, '-', label_array[k][j])

    print('--> time elapsed: ', time.time() - t_s)







