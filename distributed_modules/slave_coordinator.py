from multiprocessing.managers import BaseManager
import sys
import zipfile
from keras import backend as K
sys.path.append("..")

from utils import init_directories
from selfplay_worker import *
from train_worker import *
from evaluate_worker import *
from predicting_queue_worker import init_predicting_workers, destroy_predicting_workers
from distributed_modules.distribution_config import dconf
from distributed_modules.distribution_config import ASYNC_PIPELINE_STATE
from model import load_best_model, load_latest_model
import resource
import logging
from app_log import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

REMOTE_HOST = dconf['MASTER_IP']
REMOTE_PORT = dconf['PORT']
REMOTE_PASSWORD = dconf['REMOTE_PASSWORD']


def registerRemoteFunc():
    BaseManager.register("get_job")
    BaseManager.register("get_model")
    BaseManager.register("finish_job")
    mgr = BaseManager(address=( REMOTE_HOST, REMOTE_PORT), authkey=REMOTE_PASSWORD )
    mgr.connect()
    return mgr


def save_model(model_name, content, err):
    if err is not None:
        print("ERROR: %s" %err)
        return
    try:
        file_name = os.path.join(conf['MODEL_DIR'], model_name)
        f = open(file_name, "wb")
        f.write(content)
        f.close()
    except Exception as e:
        print("ERROR: Opening/writing file failed: %s: %s" % (file_name, e))


def model_check_update(latest_model_name, best_model_name, mgr):
    if latest_model_name == "" or best_model_name == "":
        logger.info("SKIP CHECKING MODEL BECAUSE OF EMPTY NAME %s %s", latest_model_name, best_model_name)
        return
    current_best_model = load_best_model()
    current_latest_model = load_latest_model()

    if current_best_model.name != best_model_name:
        content, err = mgr.get_model(best_model_name)
        save_model(best_model_name, content, err)
    if current_latest_model.name != latest_model_name:
        content, err = mgr.get_model(latest_model_name)
        save_model(latest_model_name, content, err)
    K.clear_session()


def zip_folder(dir, zip_ref):
    for folder, subfolders, files in os.walk(dir):
        for f in files:
            zip_ref.write(os.path.join(folder, f))


def send_finish_jobs(jobs, mgr):
    job_id = jobs['id']
    file_name = job_id + ".zip"
    zip_ref = zipfile.ZipFile(file_name, 'w')
    for dir in jobs['out_dirs']:
        # file_name = dir.split("/")[-1] + ".zip"
        zip_folder(dir, zip_ref)
    zip_ref.close()
    f = open(file_name, "rb")
    c = f.read()
    f.close()
    mgr.finish_job(job_id, c)


def main():
    init_directories()
    resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, -1))
    sys.setrecursionlimit(10 ** 6)
    GPUs = conf['GPUs']

    mgr = registerRemoteFunc()

    while True:
        jobs = mgr.get_job(concurency=len(GPUs))._getvalue()
        logger.info("GOT JOBS %", jobs)
        out_dirs = jobs['out_dirs']
        assert len(out_dirs) <= len(GPUs)
        state = jobs['state']
        model_check_update(jobs['latest_model_name'], jobs['best_model_name'], mgr)
        if state == ASYNC_PIPELINE_STATE.SELF_PLAYING:
            logger.info("STARTING REMOTE SELF_PLAY PHASE WITH %s GPUs", len(GPUs))
            init_predicting_workers(GPUs)
            workers = [NoModelSelfPlayWorker(i) for i, dir in enumerate(out_dirs)]
            for p in workers: p.start()
            for p in workers: p.join()
            workers.clear()
            send_finish_jobs(jobs, mgr)
            logger.info("FINISHED SELF_PLAY JOBS %", jobs['id'])
            destroy_predicting_workers(GPUs)
        elif state == ASYNC_PIPELINE_STATE.EVALUATING:
            logger.info("STARTING REMOTE EVALUATION PHASE WITH %s GPUs", len(GPUs))
            init_predicting_workers(GPUs)
            workers = [NoModelEvaluateWorker(i) for i in GPUs]
            for p in workers: p.start()
            for p in workers: p.join()
            workers.clear()
            send_finish_jobs(jobs, mgr)
            logger.info("FINISHED EVALUATION JOBS %", jobs["id"])
            destroy_predicting_workers(GPUs)
        else:
            print("Unhandled state %s. Sleep 5 to wait for new state" % state)
            time.sleep(5)
            continue

if __name__ == "__main__":
    main()
