import logging
import resource
import sys

from app_log import setup_logging
from distribution_config import is_slave_working, turn_on_event, ASYNC_PIPELINE_STATE
from evaluate_worker import *
from selfplay_worker import *
from train_worker import *
from utils import init_directories, clean_up_empty

setup_logging()
logger = logging.getLogger(__name__)


def main():
    init_directories()
    clean_up_empty()
    resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, -1))
    sys.setrecursionlimit(10 ** 6)
    GPUs = conf['GPUs']
    START_PHASE = "EVALUATION"
    STARTED = False

    while True:
        if STARTED or START_PHASE == "SELF-PLAY":
            STARTED = True
            logger.info("STARTING SELF_PLAY PHASE WITH %s GPUs", len(GPUs))
            turn_on_event(ASYNC_PIPELINE_STATE.SELF_PLAYING)
            workers = [SelfPlayWorker(i) for i in GPUs]
            for p in workers: p.start()
            for p in workers: p.join()
            while is_slave_working():
                time.sleep(2)
            workers.clear()
        if STARTED or START_PHASE == "TRAINING":
            STARTED = True
            logger.info("STARTING TRAINING PHASE with %s GPUs", len(GPUs))
            turn_on_event(ASYNC_PIPELINE_STATE.TRAINING)
            trainer = TrainWorker([i for i in GPUs])
            trainer.start()
            trainer.join()
        if STARTED or START_PHASE == "EVALUATION":
            STARTED = True
            logger.info("STARTING EVALUATION PHASE WITH %s GPUs", len(GPUs))
            turn_on_event(ASYNC_PIPELINE_STATE.EVALUATING)
            workers = [EvaluateWorker(i) for i in GPUs]
            for p in workers: p.start()
            for p in workers: p.join()
            while is_slave_working():
                time.sleep(2)
            workers.clear()

            promote_best_model()


if __name__ == "__main__":
    main()
