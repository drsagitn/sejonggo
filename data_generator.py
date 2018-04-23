import numpy as np
import keras
from conf import conf
import h5py
from sklearn.model_selection import train_test_split
import os
import shutil


def get_prev_self_play_model_dir(best_model_name=None):
    if best_model_name is None:  # get the latest best model
        max_index = np.Inf
    else:  # get the latest best model until this best model
        try:
            best_model_name = best_model_name.split("/")[-1]  # get the file name only if input is long path
            best_model_name = best_model_name.split('.')[0]
            max_index = int(best_model_name.split("_")[-1])
        except:
            max_index = np.Inf

    index = 0
    best_model_name_result = None
    for filename in os.listdir(conf['SELF_PLAY_DIR']):
        try:
            name = filename.split('.')[0]  # remove .h5
            i = int(name.split('_')[-1])  # may throw exception here
            if index < i < max_index:
                best_model_name_result = filename
                index = i
        except:
            continue
    return os.path.join(conf['SELF_PLAY_DIR'], best_model_name_result) if best_model_name_result else None


def clean_unused_self_play_data(latest_trained_dir):
    while latest_trained_dir is not None:
        latest_trained_dir = get_prev_self_play_model_dir(latest_trained_dir)
        if latest_trained_dir is not None:
            shutil.rmtree(latest_trained_dir)


def get_training_desc():
    # a sliding window implementation to get most recent 500,000 self-play games
    all_files = []
    n_game = 0
    self_play_best_model_dir = None
    while n_game < conf['N_MOST_RECENT_GAMES']:
        self_play_best_model_dir = get_prev_self_play_model_dir(self_play_best_model_dir)
        if self_play_best_model_dir is None:
            if n_game == 0: # Found no game data at all
                raise FileNotFoundError("Can not find self-play directory")
            break
        n_game += len(os.listdir(self_play_best_model_dir))
        for root, dirs, files in os.walk(self_play_best_model_dir):
            for f in files:
                full_filename = os.path.join(root, f)
                all_files.append(full_filename)

    x_train, x_test = train_test_split(all_files, test_size=0.1, random_state=2)
    #  clean up old data that not longer needed for training
    clean_unused_self_play_data(self_play_best_model_dir)
    return {'train': x_train, 'validation': x_test}


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_batch):
        SIZE = conf['SIZE']
        X = np.zeros((self.batch_size, *self.dim))
        policy_y = np.zeros((self.batch_size, 1))
        value_y = np.zeros((self.batch_size, SIZE * SIZE + 1))
        for j, filename in enumerate(list_IDs_batch):
            with h5py.File(filename) as f:
                board = f['board'][:]
                policy = f['policy_target'][:]
                value_target = f['value_target'][()]
                X[j] = board
                policy_y[j] = value_target
                value_y[j] = policy

        return X, [value_y, policy_y]
