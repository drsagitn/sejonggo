from predicting_queue_worker import init_predicting_workers, destroy_predicting_workers
from conf import conf
from selfplay_worker import NoModelSelfPlayWorker
from utils import init_directories, clean_up_empty
from predicting_queue_worker import put_name_request
from scpy import sync_all_game_data
import sys


def main():
    sys.setrecursionlimit(2000)
    init_directories()
    clean_up_empty()
    GPUs = conf['GPUs']
    finished_best_model_name = None
    while True:
        init_predicting_workers(GPUs)
        #  Check if we did self-play on this best model or not
        curr_best_model_name = put_name_request("BEST")
        if curr_best_model_name != finished_best_model_name:
            finished_best_model_name = curr_best_model_name
        else:
            print("No new best model for self-playing. Stopping..")
            destroy_predicting_workers()
            break

        workers = [NoModelSelfPlayWorker(i) for i in GPUs]
        for p in workers: p.start()
        for p in workers: p.join()
        destroy_predicting_workers()


if __name__ == "__main__":
    main()