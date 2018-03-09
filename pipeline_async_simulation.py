from utils import init_directories
from selfplay_worker import *
from train_worker import *
from evaluate_worker import *
from predicting_queue_worker import init_predicting_workers, destroy_predicting_workers
import resource, sys
import logging
from app_log import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    init_directories()
    resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, -1))
    sys.setrecursionlimit(10 ** 6)
    GPUs = conf['GPUs']
    START_PHASE = "SELF-PLAY"
    STARTED = False

    while True:
        if STARTED or START_PHASE == "SELF-PLAY":
            STARTED = True
            # SELF-PLAY PHASE - MULTI GPUs
            logger.info("STARTING SELF_PLAY PHASE WITH %s GPUs", len(GPUs))
            init_predicting_workers(GPUs)
            workers = [NoModelSelfPlayWorker(i) for i in GPUs]
            for p in workers: p.start()
            for p in workers: p.join()
            destroy_predicting_workers(GPUs)
            workers.clear()
        if STARTED or START_PHASE == "TRAINING":
            STARTED = True
            # # TRAINING PHASE - MULTI GPUs
            logger.info("STARTING TRAINING PHASE with %s GPUs", len(GPUs))
            trainer = TrainWorker([i for i in GPUs])
            trainer.start()
            trainer.join()
        if STARTED or START_PHASE == "EVALUATION":
            STARTED = True
            # EVALUATION PHASE - MULTI GPUs
            logger.info("STARTING EVALUATION PHASE WITH %s GPUs", len(GPUs))
            workers = [NoModelEvaluateWorker(i) for i in GPUs]
            for p in workers: p.start()
            for p in workers: p.join()
            workers.clear()

            promoter = EvaluateWorker(0, task="promote_best_model")
            promoter.start()
            promoter.join()


if __name__ == "__main__":
    main()
