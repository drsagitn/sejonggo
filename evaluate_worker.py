from multiprocessing import Process
from self_play import *
from tqdm import tqdm
import time
from pathlib import Path
from evaluator import promote_best_model
from simulation_workers import init_simulation_workers, destroy_simulation_workers
from predicting_service import init_predicting_service
from nomodel_self_play import play_game_async
from predicting_queue_worker import put_name_request, destroy_predicting_workers
setup_logging()
logger = logging.getLogger(__name__)

MCTS_SIMULATIONS = conf['MCTS_SIMULATIONS']
EVALUATE_N_GAMES = conf['EVALUATE_N_GAMES']
EVALUATE_MARGIN = conf['EVALUATE_MARGIN']
EVAL_DIR = conf['EVAL_DIR']

class EvaluateWorker(Process):
    def __init__(self, gpuid, forever=False, task="evaluate"):
        Process.__init__(self, name='EvaluateProcessor')
        self._gpuid = gpuid
        self._forever = forever
        self._task = task

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
        if self._task == "promote_best_model":
            promote_best_model()
            return
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
                    directory = os.path.join(EVAL_DIR, latest_model.name, "game_%03d" % game)
                    if os.path.isdir(directory):
                        continue
                    os.makedirs(directory)

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
            else:
                logger.info("No new trained model")
                if self._forever:
                    logger.info("Sleep for %s seconds", conf['SLEEP_SECONDS'])
                    time.sleep(conf['SLEEP_SECONDS'])
            if not self._forever:
                break
            latest_model, best_model = self.load_model()

        destroy_simulation_workers()
        K.clear_session()

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
            # set environment
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)
            init_predicting_service(self._gpuid)
            init_simulation_workers()

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
                    os.makedirs(directory)

                    start = datetime.datetime.now()
                    game_data = play_game_async("BEST_SYM", "LATEST_SYM", MCTS_SIMULATIONS, stop_exploration=0)
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

                    # save_game_data(best_model.name, game, game_data)
                    self.save_eval_game(latest_model_name, game, winner_model)

            destroy_simulation_workers()
            destroy_predicting_workers(self._gpuid)
        except Exception as e:
            print("EXCEPTION!!!: %s" % e)
