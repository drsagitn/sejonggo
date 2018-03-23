from multiprocessing.managers import BaseManager
import sys
import zipfile
from keras import backend as K

from utils import init_directories, clean_up_empty
from selfplay_worker import *
from train_worker import *
from evaluate_worker import *
from predicting_queue_worker import init_predicting_workers, destroy_predicting_workers
from distribution_config import dconf, ASYNC_PIPELINE_STATE, get_latest_model_name, get_best_model_name
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
    if latest_model_name == "" and best_model_name == "":
        logger.info("SKIP CHECKING MODEL BECAUSE OF EMPTY NAME %s %s", latest_model_name, best_model_name)
        return
    current_best_model_name = get_best_model_name()
    current_latest_model_name = get_latest_model_name()

    logger.info("CHECKING MODEL UP TO DATE")
    if current_best_model_name != best_model_name and best_model_name != "":
        logger.info("UPDATING BEST MODEL FROM MASTER %s", best_model_name)
        content, err = mgr.get_model(best_model_name)._getvalue()
        save_model(best_model_name, content, err)
        save_model(conf['BEST_MODEL'], content, err)
    if current_latest_model_name != latest_model_name and latest_model_name != "":
        logger.info("UPDATING LATEST MODEL FROM MASTER %s", best_model_name)
        content, err = mgr.get_model(latest_model_name)._getvalue()
        save_model(latest_model_name, content, err)
    logger.info("DONE.")


def zip_folder(dir, zip_ref):
    for folder, subfolders, files in os.walk(dir):
        for f in files:
            zip_ref.write(os.path.join(folder, f))


def send_finish_jobs(jobs, mgr):
    job_id = jobs['id']
    file_name = str(job_id) + ".zip"
    zip_ref = zipfile.ZipFile(file_name, 'w')
    for dir in jobs['out_dirs']:
        # file_name = dir.split("/")[-1] + ".zip"
        zip_folder(dir, zip_ref)
    zip_ref.close()
    f = open(file_name, "rb")
    c = f.read()
    f.close()
    mgr.finish_job(job_id, c)
    # should delete zip file when done

def extract_game_number(dir):
    s = dir.split("_")[-1]
    return int(s)

def main():
    init_directories()
    clean_up_empty()
    resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, -1))
    sys.setrecursionlimit(10 ** 6)
    GPUs = conf['GPUs']

    mgr = registerRemoteFunc()

    while True:
        jobs = mgr.get_job(concurency=len(GPUs))._getvalue()
        logger.info("GOT JOBS %s", jobs)
        out_dirs = jobs['out_dirs']
        assert len(out_dirs) <= len(GPUs)
        state = jobs['state']
        model_check_update(jobs['latest_model_name'], jobs['best_model_name'], mgr)
        if state == ASYNC_PIPELINE_STATE.SELF_PLAYING.name:
            logger.info("STARTING REMOTE SELF_PLAY PHASE WITH %s GPUs", len(GPUs))
            workers = [SelfPlayWorker(i, one_game_only=extract_game_number(dir)) for i, dir in enumerate(out_dirs)]
            for p in workers: p.start()
            for p in workers: p.join()
            workers.clear()
            send_finish_jobs(jobs, mgr)
            logger.info("FINISHED SELF_PLAY JOBS %", jobs['id'])
        elif state == ASYNC_PIPELINE_STATE.EVALUATING.name:
            logger.info("STARTING REMOTE EVALUATION PHASE WITH %s GPUs", len(GPUs))
            workers = [EvaluateWorker(i, one_game_only=extract_game_number(dir)) for i in GPUs]
            for p in workers: p.start()
            for p in workers: p.join()
            workers.clear()
            send_finish_jobs(jobs, mgr)
            logger.info("FINISHED EVALUATION JOBS %", jobs["id"])
        else:
            print("Unhandled state %s. Sleep 5 to wait for new state" % state)
            time.sleep(5)
            continue

if __name__ == "__main__":
    main()
