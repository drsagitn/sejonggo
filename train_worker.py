from multiprocessing import Process
from model import *
import os
from conf import conf
from train import train_by_multi_gpus
import time
setup_logging()
logger = logging.getLogger(__name__)
SIZE = conf['SIZE']
BATCH_SIZE = conf['TRAIN_BATCH_SIZE']
EPOCHS_PER_SAVE = conf['EPOCHS_PER_SAVE']
NUM_WORKERS = conf['NUM_WORKERS']
VALIDATION_SPLIT = conf['VALIDATION_SPLIT']
SELF_PLAY_DATA = conf['SELF_PLAY_DIR']
INIT_LR = 5e-3


class TrainWorker(Process):
    def __init__(self, gpuid, forever=False):
        Process.__init__(self, name='ModelProcessor')
        self._gpuid = gpuid
        self.n_gpu = len(gpuid)
        self._forever = forever

    def run(self):
        # set environment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid).strip('[').strip(']').strip(' ')
        logger.info('CUDA_VISIBLE_DEVICES %s', os.environ["CUDA_VISIBLE_DEVICES"])

        while True:
            train_by_multi_gpus(self.n_gpu)
            if self._forever:
                logger.info("Sleep %s seconds to wait for self-play", conf['SLEEP_SECONDS'])
                time.sleep(conf['SLEEP_SECONDS'])  # wait for self-play
            else:
                break

        K.clear_session()
