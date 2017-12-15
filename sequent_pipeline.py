from utils import init_directories
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
    n_gpu = conf['N_GPU']

    # while True:
    # SELF-PLAY PHASE - MULTI GPUs
    logger.info("STARTING SELF_PLAY PHASE WITH %s GPUs", n_gpu)
    workers = [SelfPlayWorker(i) for i in range(n_gpu)]
    for p in workers: p.start()
    for p in workers: p.join()
    workers.clear()

    # # TRAINING PHASE - 1 GPU
    logger.info("STARTING TRAINING PHASE with 1 GPUs")
    workers.append(TrainWorker(0))
    for p in workers: p.start()
    for p in workers: p.join()
    workers.clear()

    # EVALUATION PHASE - MULTI GPUs
    logger.info("STARTING EVALUATION PHASE WITH %s GPUs", n_gpu)
    for i in range(n_gpu):
        workers.append(EvaluateWorker(i))
    for p in workers: p.start()
    for p in workers: p.join()
    promote_best_model()
    workers.clear()


if __name__ == "__main__":
    main()
