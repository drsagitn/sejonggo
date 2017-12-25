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
from model import load_latest_model, loss, SGD, load_best_model


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
                verbose=0, batch_size=BATCH_SIZE
            )
    model.name = new_name.split('.')[0]
    model.save(os.path.join(conf['MODEL_DIR'], new_name))


def train_by_multi_gpus(n_gpu=1, epochs=None):
    import tensorflow as tf
    logger.info("Training with %s GPUs", n_gpu)
    if n_gpu > 1:
        with tf.device('/cpu:0'):
            model = load_latest_model()
    else:
        model = load_latest_model()

    with tf.device('/cpu:0'):
        best_model = load_best_model()

    base_name, index = model.name.split('_')
    new_name = "_".join([base_name, str(int(index) + 1)]) + ".h5"

    all_data_file_names = get_file_names_data_dir(os.path.join(SELF_PLAY_DATA, best_model.name))
    tf_callback = TensorBoard(log_dir=os.path.join(conf['LOG_DIR'], new_name),
                              histogram_freq=conf['HISTOGRAM_FREQ'], batch_size=BATCH_SIZE, write_graph=False,
                              write_grads=False)
    nan_callback = TerminateOnNaN()

    if n_gpu > 1:
        pmodel = multi_gpu_model(model, gpus=n_gpu)
        opt = SGD(lr=1e-2, momentum=0.9)
        pmodel.compile(loss=loss, optimizer=opt, metrics=["accuracy"])
    else:
        pmodel = model

    if epochs is None:
        epochs = EPOCHS_PER_SAVE
    for epoch in tqdm.tqdm(range(epochs), desc="Epochs"):
        for worker in tqdm.tqdm(range(NUM_WORKERS), desc="Iteration"):
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

            fake_epoch = epoch * NUM_WORKERS + worker  # used as initial_epoch, epochs is to be understood as "final epoch". The model is trained until the epoch of index epochs is reached.

            pmodel.fit(X, [value_y, policy_y],
                      initial_epoch=fake_epoch,
                      epochs=fake_epoch + 1,
                      validation_split=VALIDATION_SPLIT,  # Needed for TensorBoard histograms and gradi
                      callbacks=[tf_callback, nan_callback],
                      verbose=0, batch_size=BATCH_SIZE)

    model.name = new_name.split('.')[0]
    model.save(os.path.join(conf['MODEL_DIR'], new_name))
    logger.info("Finished training with multi GPUs. New model %s saved", new_name)
    return model


def get_file_names_data_dir(data_dir):
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            full_filename = os.path.join(root, f)
            all_files.append(full_filename)
    return all_files