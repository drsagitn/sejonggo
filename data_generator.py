import numpy as np
import keras
from conf import conf
import h5py
from sklearn.model_selection import train_test_split
import os


def get_training_desc():
    # find best model name (latest model in self-play dir)
    best_model_name = None
    index = 0
    for filename in os.listdir(conf['SELF_PLAY_DIR']):
        try:
            name = filename.split('.')[0] # remove .h5
            i = int(name.split('_')[-1]) #may throw exception here
            if i > index:
                best_model_name = filename
                index = i
        except:
            continue
    if not best_model_name:
        raise FileNotFoundError("Can not find self-play directory")
    all_files = []
    for root, dirs, files in os.walk(os.path.join(conf['SELF_PLAY_DIR'], best_model_name)):
        for f in files:
            full_filename = os.path.join(root, f)
            all_files.append(full_filename)
    x_train, x_test = train_test_split(all_files, test_size=0.1, random_state=2)
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
