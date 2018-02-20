from predicting_worker import *
import logging
from app_log import setup_logging
setup_logging()
logger = logging.getLogger(__name__)
from conf import conf
import time

predicting_workers = {}
def init_predicting_service(GPU_id):
    print("STARTING PREDICTION SERVICES WITH GPU %s" % GPU_id)
    p = PredictingWorker(GPU_id)
    predicting_workers[GPU_id] = p
    p.start()
    time.sleep(10)

def init_multiple_predicting_services(GPUs):
    for id in GPUs:
        init_multiple_predicting_services(id)

def shutdown_predicting_service(GPU_id):
    p = predicting_workers.get(GPU_id)
    if p and p.is_alive():
        print("SHUTTING DOWN PREDICTING SERVICES WITH GPU %s" % GPU_id)
        p.terminate()


if __name__ == "__main__":
    init_predicting_service(0)
    # time.sleep(30)
    # shutdown_predicting_service(0)
