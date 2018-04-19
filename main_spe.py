from predicting_queue_worker import init_predicting_workers, destroy_predicting_workers
from conf import conf
from selfplay_worker import NoModelSelfPlayWorker
from evaluate_worker import NoModelEvaluateWorker, EvaluateWorker
from utils import init_directories, clean_up_empty
from scpy import sync_all_game_data, retrieve_model
import sys


def main():
    sys.setrecursionlimit(2000)
    init_directories()
    clean_up_empty()
    sync_all_game_data(conf['SELF_PLAY_DIR'])
    GPUs = conf['GPUs']
    while True:
        # SELF-PLAY
        init_predicting_workers(GPUs)
        workers = [NoModelSelfPlayWorker(i) for i in GPUs]
        for p in workers: p.start()
        for p in workers: p.join()
        destroy_predicting_workers()

        # EVALUATE
        init_predicting_workers(GPUs) # re-init predicting worker to run with latest trained model (sent from train server)
        workers = [NoModelEvaluateWorker(i) for i in GPUs]
        for p in workers: p.start()
        for p in workers: p.join()
        workers.clear()
        destroy_predicting_workers(GPUs)

        # PROMOTE BEST MODEL
        promoter = EvaluateWorker(0, task="promote_best_model")
        promoter.start()
        promoter.join()


if __name__ == "__main__":
    main()