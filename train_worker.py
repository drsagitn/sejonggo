from multiprocessing import Process
from model import *
import os
import h5py
import numpy as np
from keras.callbacks import TensorBoard, TerminateOnNaN
from random import sample
from conf import conf
import tqdm
import time
setup_logging()
logger = logging.getLogger(__name__)
SIZE = conf['SIZE']
BATCH_SIZE = conf['TRAIN_BATCH_SIZE']
EPOCHS_PER_SAVE = conf['EPOCHS_PER_SAVE']
NUM_WORKERS = conf['NUM_WORKERS']
VALIDATION_SPLIT = conf['VALIDATION_SPLIT']


class TrainWorker(Process):
    def __init__(self, gpuid, queue=None):
        Process.__init__(self, name='ModelProcessor')
        self._gpuid = gpuid
        self._queue = queue
        self.model_self_play_data_dir = ""
        self.all_data_file_names = []

    def run(self):
        # set environment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)
        logger.info('cuda_visible_device %s', os.environ["CUDA_VISIBLE_DEVICES"])

        while True:
            # load models
            logger.info("Loading latest model...")
            model = load_latest_model()
            logger.info("Loaded latest model %s", model.name)
            self.model_self_play_data_dir = os.path.join("self_play_data", model.name)
            logger.debug("data dir %s, num of subdir %s", self.model_self_play_data_dir,
                         len(os.listdir(self.model_self_play_data_dir)))

            if len(os.listdir(self.model_self_play_data_dir)) >= conf['N_GAMES']:  # self-play finished their work and generated enough data
                self.all_data_file_names = self.get_file_names_data_dir()
                break
            else:
                logger.info("Sleep %s seconds to wait for self-play to generate %s games. Current generated %s games",
                            conf['SLEEP_SECONDS'], conf['N_GAMES'], len(os.listdir(self.model_self_play_data_dir)))
                time.sleep(conf['SLEEP_SECONDS'])  # wait for self-play

        self.train(model)
        K.clear_session()

    def train(self, model, epochs=None):
        if epochs is None:
            epochs = EPOCHS_PER_SAVE
        logger.info('Start training for %s epochs', epochs)
        name = model.name
        base_name, index = name.split('_')
        new_name = "_".join([base_name, str(int(index) + 1)]) + ".h5"
        tf_callback = TensorBoard(log_dir=os.path.join(conf['LOG_DIR'], new_name),
                                  histogram_freq=conf['HISTOGRAM_FREQ'], batch_size=BATCH_SIZE, write_graph=False,
                                  write_grads=False)
        nan_callback = TerminateOnNaN()

        for epoch in tqdm.tqdm(range(epochs), desc="Epochs"):
            values = []
            for worker in tqdm.tqdm(range(NUM_WORKERS), desc="Iteration"):
                files = sample(self.all_data_file_names, BATCH_SIZE)  # RANDOM because we use SGD (Stochastic Gradient Decent)

                X = np.zeros((BATCH_SIZE, SIZE, SIZE, 17))
                policy_y = np.zeros((BATCH_SIZE, 1))
                value_y = np.zeros((BATCH_SIZE, SIZE * SIZE + 1))
                for j, filename in enumerate(files):
                    with h5py.File(filename) as f:
                        board = f['board'][:]
                        policy = f['policy_target'][:]
                        value_target = f['value_target'][()]

                        values.append(value_target)
                        X[j] = board
                        policy_y[j] = value_target
                        value_y[j] = policy

                fake_epoch = epoch * NUM_WORKERS + worker  # For tensorboard
                model.fit(X, [value_y, policy_y],
                          initial_epoch=fake_epoch,
                          epochs=fake_epoch + 1,
                          validation_split=VALIDATION_SPLIT,  # Needed for TensorBoard histograms and gradi
                          callbacks=[tf_callback, nan_callback],
                          verbose=0,
                          )
        model.name = new_name.split('.')[0]
        save_path = os.path.join(conf['MODEL_DIR'], new_name)
        model.save(save_path)
        logger.info("Train completed. Model saved to %s", save_path)

    def get_file_names_data_dir(self):
        all_files = []
        for root, dirs, files in os.walk(self.model_self_play_data_dir):
            for f in files:
                full_filename = os.path.join(root, f)
                all_files.append(full_filename)
        return all_files



    def predict(self, xnet, imgfile):
       return None