import os
import h5py
import numpy as np
from keras.callbacks import TensorBoard, TerminateOnNaN
from random import sample
from conf import conf
import tqdm
import logging
from app_log import setup_logging
from keras.utils import multi_gpu_model
from model import *
from keras.applications import Xception

setup_logging()
logger = logging.getLogger(__name__)
SIZE = conf['SIZE']
BATCH_SIZE = conf['TRAIN_BATCH_SIZE']
EPOCHS_PER_SAVE = conf['EPOCHS_PER_SAVE']
NUM_WORKERS = conf['NUM_WORKERS']
VALIDATION_SPLIT = conf['VALIDATION_SPLIT']
INIT_LR = 5e-3
SELF_PLAY_DATA = conf['SELF_PLAY_DIR']

def train(model, game_model_name, epochs=None):
    if epochs is None:
        epochs = EPOCHS_PER_SAVE
    logger.info('train for %s epochs', epochs)
    name = model.name
    base_name, index = name.split('_')
    logger.info('model name %s', name)
    new_name = "_".join([base_name, str(int(index) + 1)]) + ".h5"
    tf_callback = TensorBoard(log_dir=os.path.join(conf['LOG_DIR'], new_name),
            histogram_freq=conf['HISTOGRAM_FREQ'], batch_size=BATCH_SIZE, write_graph=False, write_grads=False)
    nan_callback = TerminateOnNaN()

    directory = os.path.join("games", game_model_name)
    all_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.sgf'):  # Ignore sgf files.
                continue
            full_filename = os.path.join(root, f)
            all_files.append(full_filename) # IS THIS RECENT GAMES?
    for epoch in tqdm.tqdm(range(epochs), desc="Epochs"):
        values = []
        for worker in tqdm.tqdm(range(NUM_WORKERS), desc="Iteration"):
            files = sample(all_files, BATCH_SIZE)  # RANDOM because we use SGD (Stochastic Gradient Decent)

            X = np.zeros((BATCH_SIZE, SIZE, SIZE, 17))
            policy_y = np.zeros((BATCH_SIZE, 1))
            value_y = np.zeros((BATCH_SIZE, SIZE*SIZE + 1))
            for j, filename in enumerate(files):
                with h5py.File(filename) as f:
                    board = f['board'][:]
                    policy = f['policy_target'][:]
                    value_target = f['value_target'][()]

                    values.append(value_target)
                    X[j] = board
                    policy_y[j] = value_target
                    value_y[j] = policy

            fake_epoch = epoch * NUM_WORKERS + worker # For tensorboard
            model.fit(X, [value_y, policy_y],
                initial_epoch=fake_epoch,
                epochs=fake_epoch + 1,
                validation_split=VALIDATION_SPLIT, # Needed for TensorBoard histograms and gradi
                callbacks=[tf_callback, nan_callback],
                verbose=0,
            )
    model.name = new_name.split('.')[0]
    model.save(os.path.join(conf['MODEL_DIR'], new_name))

def train_multi_gpus2():
    num_samples = 1000
    height = 224
    width = 224
    num_classes = 1000
    with tf.device('/cpu:0'):
        model = Xception(weights=None,
                         input_shape=(height, width, 3),
                         classes=num_classes)
    parallel_model = multi_gpu_model(model, gpus=8)
    parallel_model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop')

    x = np.random.random((num_samples, height, width, 3))
    y = np.random.random((num_samples, num_classes))
    parallel_model.fit(x, y, epochs=20, batch_size=256)
    model.save('my_model.h5')

def train_multi_gpus(model, epochs=None, n_gpu=1):
    logger.info("Training with %s GPUs", n_gpu)
    all_data_file_names = get_file_names_data_dir(os.path.join(SELF_PLAY_DATA, model.name))
    model = multi_gpu_model(model, gpus=n_gpu)
    tf_callback = TensorBoard(log_dir=os.path.join(conf['LOG_DIR'], "multi_gpu_model"),
                              histogram_freq=conf['HISTOGRAM_FREQ'], batch_size=BATCH_SIZE, write_graph=False,
                              write_grads=False)
    nan_callback = TerminateOnNaN()
    files = sample(all_data_file_names, BATCH_SIZE)  # RANDOM because we use SGD (Stochastic Gradient Decent)

    X = np.zeros((BATCH_SIZE, SIZE, SIZE, 17))
    policy_y = np.zeros((BATCH_SIZE, 1))
    value_y = np.zeros((BATCH_SIZE, SIZE * SIZE + 1))
    for j, filename in enumerate(files):
        with h5py.File(filename) as f:
            board = f['board'][:]
            policy = f['policy_target'][:]
            value_target = f['value_target'][()]
            X[j] = board
            policy_y[j] = value_target
            value_y[j] = policy

    fake_epoch = 1  # For tensorboard
    opt = SGD(lr=INIT_LR, momentum=0.9)
    model.compile(loss=loss, optimizer=opt,
                  metrics=["accuracy"])
    model.fit(X, [value_y, policy_y],
              initial_epoch=fake_epoch,
              epochs=fake_epoch + 1,
              validation_split=VALIDATION_SPLIT,  # Needed for TensorBoard histograms and gradi
              callbacks=[tf_callback, nan_callback],
              verbose=0,
              )
    model.save(os.path.join(conf['MODEL_DIR'], "multi_gpu_model"))
    logger.info("Finished training with multi gpus")


def get_file_names_data_dir(data_dir):
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            full_filename = os.path.join(root, f)
            all_files.append(full_filename)
    return all_files