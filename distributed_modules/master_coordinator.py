import sys
import os
import time
import zipfile
sys.path.append("..")
from multiprocessing.managers import BaseManager
from distributed_modules.distribution_config import self_play_event, evaluate_event, all_slaves_finished_event
from conf import conf
from distributed_modules.distribution_config import dconf
from distributed_modules.distribution_config import ASYNC_PIPELINE_STATE
from predicting_queue_worker import put_name_request
import logging
from app_log import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

PORT = dconf['PORT']
REMOTE_PASSWORD = dconf['REMOTE_PASSWORD']


def get_file(file_name):
    print("File requested: %s" % file_name)

    if not os.path.exists(file_name):
        return None, "REMOTE SERVER ERROR: File does not exist: %s" % file_name

    if os.path.isdir(file_name):
        return None, "REMOTE SERVER ERROR: Path to file expected, but path is a folder: %s" % file_name

    try:
        f = open(file_name, "rb")
        c = f.read()
        f.close()
        return c, None
    except Exception as e:
        return None, "REMOTE SERVER ERROR: Opening/reading file failed: %s: %s" % (file_name, e)


def put_file(file_name, file_contents):
    try:
        f = open(file_name, "wb")
        f.write(file_contents)
        f.close()
        return None
    except Exception as e:
        return "REMOTE SERVER ERROR: Opening/writing file failed: %s: %s" % (file_name, e)


def get_state():
    if self_play_event.is_set():
        return ASYNC_PIPELINE_STATE.SELF_PLAYING
    if evaluate_event.is_set():
        return ASYNC_PIPELINE_STATE.EVALUATING
    return ASYNC_PIPELINE_STATE.OTHERS


def get_model(model_name):
    model_dir = conf['MODEL_DIR']
    if model_name == "BEST_MODEL":
        model_name = conf['BEST_MODEL']
    model_path = os.path.join(model_dir, model_name)
    return get_file(model_path)


def _reserve_directory(dir, number):
    while number >= 0:
        directory = os.path.join(dir, "game_%05d" % number)
        print("Reserving dir %s" % directory)
        if os.path.isdir(directory):
            number -= 1
            continue
        try:
            os.makedirs(directory)
            break
        except Exception:
            number -= 1
            continue
    return directory


def get_parent_dir(dir):
    return os.path.dirname(os.path.normpath(dir))


def finish_job(job_id, result_zipfile_content):
    j = assigned_jobs.pop(job_id, None)
    if j is not None:
        try:
            # Save remote job zipfile result
            zip_file = os.path.join(conf['TMP_DIR'], job_id + ".zip")
            put_file(zip_file, result_zipfile_content)
            # Extract zip to target folder
            zip_ref = zipfile.ZipFile(zip_file, 'r')
            zip_ref.extractall(get_parent_dir(j[dir]))
            zip_ref.close()
            if len(assigned_jobs) == 0:
                logger.info("GOT JOB FINISH FROM SLAVE. UPDATE JOB LIST %", assigned_jobs)
                logger.info("CLEAN UP TEMP FOLDER!!")
                import shutil
                shutil.rmtree(conf['TMP_DIR'])
                os.makedirs(conf['TMP_DIR'])
                all_slaves_finished_event.set()
        except Exception as e:
            print("Error while saving remote job result!")
            # TODO: clear the directory
            if os.path.isdir(j['dir']):
                print("Clean result directory %s", j['dir'])
                import shutil
                shutil.rmtree(j['dir'])


assigned_jobs = {}
def get_job(concurency=1):
    state = get_state()
    directories = []
    latest_model_name = ""
    best_model_name = ""
    if state == ASYNC_PIPELINE_STATE.SELF_PLAYING:
        SELF_PLAY_DATA = conf['SELF_PLAY_DIR']
        best_model_name = put_name_request("BEST_NAME")
        n_games = conf['N_GAMES']
        for i in range(concurency):
            directory = _reserve_directory(os.path.join(SELF_PLAY_DATA, best_model_name), n_games - 1 - i)
            directories.append(directory)

    elif state == ASYNC_PIPELINE_STATE.EVALUATING:
        EVALUATE_N_GAMES = conf['EVALUATE_N_GAMES']
        best_model_name = put_name_request("BEST_NAME")
        latest_model_name = put_name_request("LATEST_NAME")
        n_games = conf['EVALUATE_N_GAMES']
        for i in range(concurency):
            directory = _reserve_directory(os.path.join(EVALUATE_N_GAMES, latest_model_name), n_games - 1 - i)
            directories.append(directory)

    job_id = int(time.time())
    job = {
        'id': job_id,
        'state': state,
        'out_dirs': directories,
        'best_model_name': best_model_name,
        'latest_model_name': latest_model_name
    }
    assigned_jobs[job_id] = job
    all_slaves_finished_event.clear()
    logger.info("ASSIGN NEW JOB. UPDATE JOB LIST %s", assigned_jobs)
    return job


def start_server( port, password ):
    print("Listening for incoming connections...")
    mgr = BaseManager( address=( '', port ), authkey=password )
    s = mgr.get_server()
    s.serve_forever()


if __name__ == "__main__":
    BaseManager.register("get_file", get_file)
    BaseManager.register("put_file", put_file)
    BaseManager.register("get_job", get_job)
    BaseManager.register("get_model", get_model)
    BaseManager.register("get_state", get_state)
    start_server(PORT, REMOTE_PASSWORD)