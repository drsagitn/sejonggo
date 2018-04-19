from multiprocessing import Process
from model import load_best_model, load_latest_model
from tqdm import tqdm
import time
import datetime
import os
from pathlib import Path
from self_play import play_game
from evaluator import promote_best_model
from simulation_workers import init_simulation_workers, destroy_simulation_workers, init_simulation_workers_by_gpuid
from nomodel_self_play import play_game_async
from predicting_queue_worker import put_name_request
from scpy import sync_game_data
from conf import conf
from sgfsave import save_game_data
from app_log import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)

MCTS_SIMULATIONS = conf['MCTS_SIMULATIONS']
EVALUATE_N_GAMES = conf['EVALUATE_N_GAMES']
EVALUATE_MARGIN = conf['EVALUATE_MARGIN']
EVAL_DIR = conf['EVAL_DIR']

class EvaluateWorker(Process):
    def __init__(self, gpuid, forever=False, one_game_only=-1):
        Process.__init__(self, name='EvaluateProcessor')
        self._gpuid = gpuid
        self._forever = forever
        self._one_game_only = one_game_only

    def load_model(self):
        best_model = load_best_model()
        logger.info("Loaded best model %s", best_model.name)

        latest_model = load_latest_model()
        logger.info("Loaded latest %s", latest_model.name)
        return latest_model, best_model

    def save_eval_game(self, model_name, game_no, winner_model):
        filepath = os.path.join(EVAL_DIR, model_name, "game_%03d" % game_no, winner_model)
        Path(filepath).touch()  # can use os.mknod(filepath) but will throw exception if file existed

    def run(self):
        # set environment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)
        logger.info('cuda_visible_device %s', os.environ["CUDA_VISIBLE_DEVICES"])
        if conf['THREAD_SIMULATION']:
            init_simulation_workers()

        # load models
        latest_model, best_model = self.load_model()

        while True:
            if latest_model.name != best_model.name:
                total = 0
                wins = 0
                desc = "Evaluation %s vs %s" % (latest_model.name, best_model.name)
                tq = tqdm(range(EVALUATE_N_GAMES), desc=desc)
                for game in tq:
                    if self._one_game_only >= 0 and tq != game:
                        continue
                    directory = os.path.join(EVAL_DIR, latest_model.name, "game_%03d" % game)
                    if os.path.isdir(directory):
                        continue
                    try:
                        os.makedirs(directory)
                    except Exception:
                        continue

                    start = datetime.datetime.now()
                    game_data = play_game(best_model, latest_model, MCTS_SIMULATIONS, stop_exploration=0)
                    stop = datetime.datetime.now()

                    # Some statistics
                    winner_model = game_data['winner_model']
                    if winner_model == latest_model.name:
                        wins += 1
                    total += 1
                    moves = len(game_data['moves'])
                    new_desc = desc + " (winrate:%s%% %.2fs/move)" % (
                    int(wins / total * 100), (stop - start).seconds / moves)
                    tq.set_description(new_desc)

                    # save_game_data(best_model.name, game, game_data)
                    self.save_eval_game(latest_model.name, game, winner_model)
                    if self._one_game_only >= 0:
                        break
            else:
                logger.info("No new trained model")
                if self._forever:
                    logger.info("Sleep for %s seconds", conf['SLEEP_SECONDS'])
                    time.sleep(conf['SLEEP_SECONDS'])
            if not self._forever:
                break
            latest_model, best_model = self.load_model()

        destroy_simulation_workers()

class NoModelEvaluateWorker(Process):
    def __init__(self, gpuid, task="evaluate"):
        Process.__init__(self, name='EvaluateProcessor')
        self._gpuid = gpuid
        self._task = task

    def save_eval_game(self, model_name, game_no, winner_model):
        filepath = os.path.join(EVAL_DIR, model_name, "game_%03d" % game_no, winner_model)
        Path(filepath).touch()  # can use os.mknod(filepath) but will throw exception if file existed

    def run(self):
        if self._task == "promote_best_model":
            promote_best_model()
            return
        try:
            init_simulation_workers_by_gpuid(self._gpuid)

            best_model_name = put_name_request("BEST_NAME")
            latest_model_name = put_name_request("LATEST_NAME")

            if latest_model_name != best_model_name:
                total = 0
                wins = 0
                desc = "Evaluation %s vs %s" % (latest_model_name, best_model_name)
                tq = tqdm(range(EVALUATE_N_GAMES), desc=desc)
                for game in tq:
                    directory = os.path.join(EVAL_DIR, latest_model_name, "game_%03d" % game)
                    if os.path.isdir(directory):
                        continue
                    try:
                        os.makedirs(directory)
                    except Exception:
                        continue

                    start = datetime.datetime.now()
                    game_data = play_game_async("BEST_SYM", "LATEST_SYM", MCTS_SIMULATIONS, stop_exploration=0, gpuid=self._gpuid)
                    stop = datetime.datetime.now()

                    # Some statistics
                    winner_model = game_data['winner_model']
                    if winner_model == latest_model_name:
                        wins += 1
                    total += 1
                    moves = len(game_data['moves'])
                    new_desc = desc + " (winrate:%s%% %.2fs/move)" % (
                    int(wins / total * 100), (stop - start).seconds / moves)
                    tq.set_description(new_desc)

                    game_name = "eval_" + str(game)
                    save_game_data(latest_model_name, game_name, game_data)  # save game data to self-play folder for training
                    self.save_eval_game(latest_model_name, game, winner_model)  # save game result for statistic
                    if conf['TRAINING_SERVER']:
                        sync_game_data(conf['SELF_PLAY_DIR'], latest_model_name, game_name)
            else:
                print("BEST MODEL and LAST MODEL are the same!! Quitting")
            destroy_simulation_workers()
        except Exception as e:
            print("EXCEPTION IN NO MODEL EVALUATION WORKER!!!: %s" % e)
