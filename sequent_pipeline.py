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
    START_PHASE = "SELF-PLAY"
    STARTED = False

    while True:
        if STARTED or START_PHASE == "SELF-PLAY":
            STARTED = True
            # SELF-PLAY PHASE - MULTI GPUs
            logger.info("STARTING SELF_PLAY PHASE WITH %s GPUs", n_gpu)
            workers = [SelfPlayWorker(i) for i in range(n_gpu)]
            for p in workers: p.start()
            for p in workers: p.join()
            workers.clear()
        if STARTED or START_PHASE == "TRAINING":
            STARTED = True
            # # TRAINING PHASE - MULTI GPUs
            logger.info("STARTING TRAINING PHASE with %s GPUs", n_gpu)
            trainer = TrainWorker([i for i in range(n_gpu)])
            trainer.start()
            trainer.join()
        if STARTED or START_PHASE == "EVALUATION":
            STARTED = True
            # EVALUATION PHASE - MULTI GPUs
            logger.info("STARTING EVALUATION PHASE WITH %s GPUs", n_gpu)
            workers = [EvaluateWorker(i) for i in range(n_gpu)]
            for p in workers: p.start()
            for p in workers: p.join()
            workers.clear()

            promoter = EvaluateWorker(0, task="promote_best_model")
            promoter.start()
            promoter.join()


if __name__ == "__main__":
    main()
