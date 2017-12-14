from utils import get_available_gpus, init_directories
from evaluator import promote_best_model
from selfplay_worker import *
from train_worker import *
from evaluate_worker import *
import logging
from app_log import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


def start_process_list(process_list):
    for worker in process_list:
        worker.start()


# def wait_for_process_list(process_list):



def start_and_wait(process_list):
    start_process_list(process_list)
    # wait_for_process_list(process_list)


def main():
    init_directories()
    gpus = get_available_gpus()
    n_gpu = len(gpus)

    workers = list()
    while True:
        # SELF-PLAY PHASE - MULTI GPUs
        logger.info("STARTING SELF_PLAY PHASE WITH %s GPUs", n_gpu)
        for i in range(n_gpu):
            workers.append(SelfPlayWorker(i))
        start_process_list(workers)
        for worker in workers:
            worker.join()
        # workers.clear()
        #
        # # TRAINING PHASE - 1 GPU
        # logger.info("STARTING TRAINING PHASE with 1 GPUs")
        # workers.append(TrainWorker(0))
        # start_and_wait(workers)
        # workers.clear()
        #
        # # EVALUATION PHASE - MULTI GPUs
        # logger.info("STARTING EVALUATION PHASE WITH %s GPUs", n_gpu)
        # for i in range(n_gpu):
        #     workers.append(EvaluateWorker(i))
        # start_and_wait(workers)
        # promote_best_model()
        # workers.clear()


if __name__ == "__main__":
    main()
