import librosa
import yaml
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import optimizers
import h5py
import random
import math
import os


# UTILS
def load_parameters():
    """
    Load the parameter set from the yaml file and return as a python dictionary.

    Change the parameters

    Returns:
        dictionary holding parameter values
    """
    return yaml.load(open('parameters.yaml'), Loader=yaml.SafeLoader)
  #hop_size = 2048
  #sample_rate = 44100

# FEATURE EXTRACTION

    """

    snippet should be 45 seconds
    
    after the melspectogram pad mel 3 seconds on the left and on the right


    """

def extract_feature(audio_path, parameters):
    """
    Extracts the mel-spectrogram from an audio file. If the file is longer than parameters['snippet_length_sec'], it
    will be trimmed. If it is shorter, it will be padded with silence



    Args:
        audio_path: Path to the audio file.
        parameters: Dictionary with system parameters.

    Returns:
        2D numpy feature matrix.
    """

    # read the audio file and if necessary resample to parameters['fs']
    samples, _ = librosa.load(audio_path, mono=True, sr=parameters['fs'])

    # normalize
    samples /= max(abs(samples))

    # either pad or cut to desired length
    num_samples = parameters['snippet_length_sec'] * parameters['fs']  # desired length in samples
    if samples.shape[0] < num_samples:
        samples = np.pad(samples, (0, num_samples - samples.shape[0]), mode='constant')  # pad with zeros
    else:
        samples = samples[:num_samples]

    # compute mel-spectrogram
    mel = librosa.feature.melspectrogram(samples,
                                         sr=parameters['fs'],
                                         n_mels=parameters['num_mel'],
                                         hop_length=parameters['hop_size'],
                                         n_fft=parameters['window_size'])

    # perceptual weighting (adjust magnitudes to human perception)
    mel = librosa.perceptual_weighting(mel, librosa.mel_frequencies(parameters['num_mel']))

    # threshold low values
    mel[mel < parameters['min_db']] = parameters['min_db']

    # scale
    mel /= abs(parameters['min_db']) # lowest value will be -min_db


    #PADDING +/- 3s

    num_samples_3sec = parameters['snippet_length_3sec'] * parameters['fs']/parameters['window_size']  # desired length in samples
    #round it down 
    pad_value = math.floor(num_samples_3sec)
    print("mel before", mel.shape);
    mel = np.pad(mel, ((0,0),(pad_value, pad_value)), 'constant', constant_values = (-1,-1))  
    #is it 0,0 or is it what it corresponds to -40dB
    print("mel", mel.shape);

    return mel



# NETWORK
def generate_network(parameters):
    """
    Generates a keras model with convolutional, pooling and dense layers.

    Args:
        parameters: Dictionary with system parameters.

    Returns:
        keras model object.
    """
    # pre-compute "image" height and width
    image_height = parameters['num_mel']
    # time to frames
    # image_width = int(np.round(parameters['snippet_length_sec'] * parameters['fs'] / float(parameters['hop_size']))) + 2*math.floor(parameters['snippet_length_3sec'] * parameters['fs']/parameters['window_size'])

    ### ValueError: Error when checking input: expected conv2d_1_input to have shape (128, 1097, 1) but got array with shape (128, 128, 1)

    image_width = 128 
    # ->>
    ### ValueError: Error when checking target: expected dense_1 to have 2 dimensions, but got array with shape (16, 61, 1)
    #### How come?? It is flattened before being sent to dense!


    # standard
    conv = Sequential()
    conv.add(Conv2D(16, (3, 3), activation='relu', input_shape=((image_height, image_width, 1))))
    conv.add(MaxPooling2D((2, 2)))
    conv.add(Dropout(0.4))
    conv.add(Conv2D(16, (3, 3), activation='relu'))
    conv.add(MaxPooling2D((2, 2)))
    conv.add(Dropout(0.4))
    conv.add(Flatten())
    conv.add(Dense(1, activation='linear'))
    opt = optimizers.Adam(lr=parameters['initial_lr'])
    conv.compile(loss='mean_squared_error', optimizer=opt)
    return conv

def data_generator(dataset, num_steps, shuffle, h5_path, parameters):
    """
    Data generator for training: Supplies the train method with features and labels taken from the hdf5 file

    Args:
        dataset: "train" or "test".
        num_steps: number of generation steps.
        shuffle: whether or not to shuffle the data
        h5_path: path to database .h5 file
        parameters: parameter dictionary

    Returns:
        feature data (x_data) and labels (y)
    """
    hdf5_file = h5py.File(h5_path, "r")  # open hdf5 file in read mode
    # point to the correct feature and label dataset
    feature_dataset = dataset + '_features' # hdf5_file[feature_dataset]
    label_dataset = dataset + '_labels' # hdf5_file[label_dataset]


    features = hdf5_file[feature_dataset] # print = <HDF5 dataset "train_features": shape (682, 128, 1097), type "<f8">
    labels = hdf5_file[label_dataset]

    #3 second padding in samples
    image_padding_amount = math.floor(parameters['snippet_length_3sec'] * parameters['fs']/parameters['hop_size']) 

    #the beginning 14,5 second in samples which is not annotated
    unannotated_beginning = math.floor(14.5 * parameters['fs']/parameters['hop_size']) 

    # Works - gets rid of the padding
    image_width_per_half_sec = int(np.round(0.5*parameters['fs']/parameters['hop_size']))

    # in order to divide subfeatures
    labels_per_audio = 61

    '''
        pseudocode:

    for feature in hdf5_file:
        for sub_feature in feature.reshape(): <- reshape according to image dimensions
            features.append(sub_feature)

    '''



    # import pdb; pdb.set_trace() # This code will pause the program at this specific line, then you can interact with the variables above. To finish, type 'c' and hit enter.
    # breakpoint()


    '''
    #################### Nadine ###########################################

    num_songs = len(features)
    images = []
    image_labels = []

    for i in range(num_songs):
        song_labels = labels[i]
        song_feature = features[i]

        for ii in range(song_labels.shape[0]):
            frame_idx = get_frame_for_label_index(ii, parameters)
            image = song_feature[:, frame_idx - image_padding_amount : frame_idx + image_padding_amount]
            label = song_labels[ii]
            images.append[image]
            labels.append[label]
    '''

    ############if we do this it would take so much memory -- instead of actually extracting features we should just make a list        


    indices = []

    num_songs = len(features)

    # these values?
    print(num_songs)
    print(labels.shape[0])
    print(labels)

    for i in range(num_songs):
        song_labels = labels[i]
        # print(song_labels.shape[0]) # 61
        for ii in range(song_labels.shape[0]):
            indices.append([i, ii]) # we don't append feature we just append the index
                                    # index of the song, index of the label


    ########################################################################

    inds = list(range(len(indices)))  # needed for shuffling


    while 1: 
        if shuffle: 
            random.shuffle(inds)
        for i in range(num_steps):
            x_data = []
            y = []
            # one loop per batch
            for ii in range(parameters['batch_size']):

                f_ind = i * parameters['batch_size'] + ii


                idx = indices[inds[f_ind]] # randomly selected pair of song index and label index

                song_feature = features[idx[0], ...] # idx[0] would be song index
                frame_idx = get_frame_for_label_index(idx[1], parameters) # idx[1] is the label index
                feature = song_feature[:, frame_idx - image_padding_amount : frame_idx + image_padding_amount]
                x_data.append(feature)

                song_labels = labels[idx[0], ...]
                label = song_labels[idx[1]]
                y.append(label)

            # convert to arrays
            x_data = np.asarray(x_data)
            y = np.asarray(y)
            # conv layers need image data format
            x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
            # send data at the end of each batch
            #print(x_data.shape)
            yield x_data, y
    ######################################################################
    # run on your machine for half an hour - if that doesn't crash we'


def get_frame_for_label_index(label_pos, parameters):
    #calculate the frame on the spectogram corresponding to label 

    # math.floor or int ???????????? 
    frame_pos = math.floor((3 + 15 + label_pos*0.5)*parameters['fs']/parameters['hop_size'])

    return frame_pos


'''
    time = 15.0 + label_index * 0.5
    feature_idx = time * parameters['fs'] / float(parameters['hop_size'])  +  for onset in onsets:
           print(‘onset: ’, onset)
'''
