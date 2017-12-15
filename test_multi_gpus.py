from utils import init_directories
from train_worker import *
from selfplay_worker import *
from evaluate_worker import *
from evaluator import promote_best_model
import logging
from app_log import setup_logging
from train import train_multi_gpus
setup_logging()
logger = logging.getLogger(__name__)


def main():
    init_directories()
    n_gpu = conf['N_GPU']
    # workers = list()
    # workers.append(TrainWorker([i for i in range(n_gpu)]))
    # for p in workers: p.start()
    # for p in workers: p.join()
    # workers.clear()

    model = load_best_model()
    train_multi_gpus(model)



if __name__ == "__main__":
    main()
