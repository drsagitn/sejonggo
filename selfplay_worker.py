from multiprocessing import Process
import time
from self_play import *
setup_logging()
logger = logging.getLogger(__name__)
SIZE = conf['SIZE']
BATCH_SIZE = conf['TRAIN_BATCH_SIZE']
EPOCHS_PER_SAVE = conf['EPOCHS_PER_SAVE']
NUM_WORKERS = conf['NUM_WORKERS']
VALIDATION_SPLIT = conf['VALIDATION_SPLIT']


class SelfPlayWorker(Process):
    def __init__(self, gpuid, forever=False):
        Process.__init__(self, name='SelfPlayProcessor')
        self._gpuid = gpuid
        self._forever = forever

    def run(self):
        # set enviornment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)
        logger.info('cuda_visible_device %s', os.environ["CUDA_VISIBLE_DEVICES"])

        name = ""
        # load models
        logger.info("Loading best model...")
        model = load_best_model()
        logger.info("Loaded %s", model.name)
        while True:
            if model.name != name:
                name = model.name
                model_self_play(model)
            else:
                logger.info("No new best model, sleep for %s seconds", conf['SLEEP_SECONDS'])
                time.sleep(conf['SLEEP_SECONDS'])
            if not self._forever:
                break
            logger.info("Loading next best model...")
            model = load_best_model()
            logger.info("Loaded %s", model.name)

        K.clear_session()