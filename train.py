"""
STEP 2: Train a DNN to classify sounds as either 'cello' or 'applause'. This is a toy example and we are only training
for 5 epochs. Validation accuracy should go above 95%.
"""

try:
    from code.feature_extraction import *
except ImportError:
    from feature_extraction import *   # when running from terminal, the directory may not be identified as a package
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os


# make sure we have the hdf5 data file
hdf5_path = '../data/dataset.hdf5'
if not os.path.isfile:
    print('ERROR: HDF5-file not found! Run create_hdf5_dataset.py first!')
    exit(0)

# load parameters
parameters = load_parameters()

# generate CNN model
model = generate_network(parameters)
model.summary()  # print a summary of the model

# pre-compute number of steps
hdf5_file = h5py.File(hdf5_path, "r")

print(len(hdf5_file['train_labels']))
num_train_steps = int(np.floor(len(hdf5_file['train_labels']) *61 / parameters['batch_size']))   ## times the no of labels per song
num_val_steps = int(np.floor(len(hdf5_file['test_labels']) *61 / parameters['batch_size']))

# callbacks
# save the best performing model
save_best = ModelCheckpoint('trained_model.h5', monitor='val_loss', save_best_only=True)
# keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

# train
model.fit_generator(data_generator('train', num_train_steps, True, hdf5_path, parameters),
                    steps_per_epoch=num_train_steps,
                    epochs=parameters['epochs'],
                    validation_data=data_generator('test', num_val_steps, False, hdf5_path, parameters),
                    validation_steps=num_val_steps,
                    callbacks=[save_best, early_stop])
