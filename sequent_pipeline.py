from utils import get_available_gpus, init_directories, start_and_wait
from evaluator import promote_best_model
from selfplay_worker import *
from train_worker import *
from evaluate_worker import *
import logging
from app_log import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    init_directories()
    gpus = get_available_gpus()
    n_gpu = len(gpus)

    workers = list()
    while True:
        # SELF-PLAY PHASE - MULTI GPUs
        logger.info("Starting Self-play Phase with %s GPUs", n_gpu)
        for i in range(n_gpu):
            workers.append(SelfPlayWorker(i))
        start_and_wait(workers)
        workers.clear()

        # TRAINING PHASE - 1 GPU
        logger.info("Starting Self-play Phase with 1 GPUs")
        workers.append(TrainWorker(0))
        start_and_wait(workers)
        workers.clear()

        # EVALUATION PHASE - MULTI GPUs
        logger.info("Starting Self-play Phase with %s GPUs", n_gpu)
        for i in range(n_gpu):
            workers.append(EvaluateWorker(i))
        start_and_wait(workers)
        promote_best_model()
        workers.clear()


if __name__ == "__main__":
    main()
