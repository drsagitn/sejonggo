from multiprocessing import Process
import time
from nomodel_self_play import play_game_async
from self_play import *
from predicting_queue_worker import put_name_request
from simulation_workers import init_simulation_workers, init_simulation_workers_by_gpuid, destroy_simulation_workers
import traceback
import sys
setup_logging()
logger = logging.getLogger(__name__)


class SelfPlayWorker(Process):
    def __init__(self, gpuid, forever=False, one_game_only=-1):
        Process.__init__(self, name='SelfPlayProcessor')
        self._gpuid = gpuid
        self._forever = forever
        self._one_game_only = one_game_only

    def run(self):
        # set environment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)
        logger.info('cuda_visible_device %s', os.environ["CUDA_VISIBLE_DEVICES"])
        if conf['THREAD_SIMULATION']:
            init_simulation_workers()

        name = ""
        model = load_best_model()
        logger.info("Loaded %s", model.name)
        while True:
            if model.name != name:
                name = model.name
                model_self_play(model, one_game_only=self._one_game_only)
            else:
                logger.info("No new best model")
                if self._forever:
                    logger.info("Sleep for %s seconds", conf['SLEEP_SECONDS'])
                    time.sleep(conf['SLEEP_SECONDS'])
            if not self._forever:
                break
            logger.info("Loading next best model...")
            model = load_best_model()
            logger.info("Loaded %s", model.name)

        destroy_simulation_workers()
        K.clear_session()


class NoModelSelfPlayWorker(Process):
    def __init__(self, gpuid):
        Process.__init__(self, name='SelfPlayProcessor')
        self._gpuid = gpuid

    def run(self):
        try:
            init_simulation_workers_by_gpuid(self._gpuid)

            n_games = conf['N_GAMES']
            game_range = conf['GAME_RANGE']
            energy = conf['ENERGY']
            model_name = put_name_request("BEST_NAME")

            desc = "Async Self play %s for %s games" % (model_name, n_games)
            games = tqdm.tqdm(range(n_games), desc=desc)
            current_resign = None
            min_values = []
            for game in range(game_range[0], game_range[1]):
                directory = os.path.join(SELF_PLAY_DATA, model_name, "game_%05d" % game)
                if os.path.isdir(directory):
                    continue
                try:
                    os.makedirs(directory)
                except Exception:
                    continue

                if random() > RESIGNATION_PERCENT:
                    resign = current_resign
                else:
                    resign = None
                start = datetime.datetime.now()
                game_data = play_game_async("BEST_SYM", "BEST_SYM", energy, conf['STOP_EXPLORATION'], gpuid=self._gpuid, self_play=True,
                                            resign_model1=resign, resign_model2=resign)
                stop = datetime.datetime.now()

                # If we did not use resignation, we had the result towards resign value.
                if resign == None:
                    winner = game_data['winner']
                    if winner == 1:
                        min_value = min([move['value'] for move in game_data['moves'][::2]])
                    else:
                        min_value = min([move['value'] for move in game_data['moves'][1::2]])
                    min_values.append(min_value)
                    l = len(min_values)
                    resignation_index = int(RESIGNATION_ALLOWED_ERROR * l)
                    if resignation_index > 0:
                        current_resign = min_values[resignation_index]

                moves = len(game_data['moves'])
                speed = ((stop - start).seconds / moves) if moves else 0.
                games.set_description(desc + " %s moves %.2fs/move " % (moves, speed))
                save_self_play_data(model_name, game, game_data)
                logger.info("Finish self-play game %s", game)
        except Exception as e:
            print("EXCEPTION in NoModelSelfPlayWorker!!!: %s" % e)
            traceback.print_exc(file=sys.stdout)
        finally:
            destroy_simulation_workers()
