from predicting_queue_worker import init_predicting_workers, destroy_predicting_workers
from conf import conf
from selfplay_worker import NoModelSelfPlayWorker


def main():
    # self-play and evaluate new trained model
    GPUs = conf['GPUs']
    # SELF-PLAY
    init_predicting_workers(GPUs)
    workers = [NoModelSelfPlayWorker(i) for i in GPUs]
    for p in workers: p.start()
    for p in workers: p.join()
    destroy_predicting_workers()

    # EVALUATE
    # get latest trained model from training server


if __name__ == "__main__":
    main()