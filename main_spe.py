import sys
from predicting_queue_worker import init_predicting_workers, destroy_predicting_workers
from conf import conf
from selfplay_worker import NoModelSelfPlayWorker
from evaluate_worker import NoModelEvaluateWorker
from utils import init_directories, clean_up_empty
from evaluator import promote_best_model


def main():
    sys.setrecursionlimit(2000)
    init_directories()
    clean_up_empty()
    GPUs = conf['GPUs']
    START_PHASE = "EVALUATING"
    while True:
        if START_PHASE != "EVALUATING":
            # SELF-PLAY
            init_predicting_workers(GPUs)
            workers = [NoModelSelfPlayWorker(i) for i in GPUs]
            for p in workers: p.start()
            for p in workers: p.join()
            destroy_predicting_workers(GPUs)

        # EVALUATE
        init_predicting_workers(GPUs)  # re-init predicting worker to run with latest trained model (sent from train server)
        workers = [NoModelEvaluateWorker(i) for i in GPUs]
        for p in workers: p.start()
        for p in workers: p.join()
        workers.clear()
        destroy_predicting_workers(GPUs)

        if promote_best_model():
            START_PHASE = ""  # there are new best model so we doing self-play in next loop


if __name__ == "__main__":
    main()