from self_play import play_game, self_play, save_game_data
from conf import conf
import os
from tqdm import tqdm
import datetime
import shutil
from scpy import sync_model
import logging
from app_log import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

MCTS_SIMULATIONS = conf['MCTS_SIMULATIONS']
EVALUATE_N_GAMES = conf['EVALUATE_N_GAMES']
EVALUATE_MARGIN = conf['EVALUATE_MARGIN']
EVAL_DIR = conf['EVAL_DIR']

def elect_model_as_best_model(model):
    self_play(model, n_games=conf['N_GAMES'], mcts_simulations=conf['MCTS_SIMULATIONS'])
    full_filename = os.path.join(conf['MODEL_DIR'], conf['BEST_MODEL'])
    model.save(full_filename)

def evaluate(best_model, tested_model):
    total = 0
    wins = 0
    desc = "Evaluation %s vs %s" % (tested_model.name, best_model.name)
    tq = tqdm(range(EVALUATE_N_GAMES), desc=desc)
    for game in tq:
        start = datetime.datetime.now()
        game_data = play_game(best_model, tested_model, MCTS_SIMULATIONS, stop_exploration=0)
        stop = datetime.datetime.now()

        winner_model = game_data['winner_model']
        if winner_model == tested_model.name:
            wins += 1
        total += 1
        moves = len(game_data['moves'])
        new_desc = desc + " (winrate:%s%% %.2fs/move)" % (int(wins/total*100), (stop - start).seconds / moves)
        tq.set_description(new_desc)

        save_game_data(best_model.name, game, game_data)

    if wins/total > EVALUATE_MARGIN:
        print("We found a new best model : %s!" % tested_model.name)
        elect_model_as_best_model(tested_model)
        return True
    return False


def eval_statistic():
    result = {}
    for model_name in os.listdir(EVAL_DIR):
        wins = 0
        total = 0
        model_dir = os.path.join(EVAL_DIR, model_name)
        if os.path.isdir(model_dir):
            for game_dir in os.listdir(model_dir):
                if game_dir.startswith('game'):  # only statistic on game_xxx folder
                    total += 1
                    if os.path.isfile(os.path.join(model_dir, game_dir, model_name)):
                        wins += 1
        result[model_name] = wins/total if total != 0 else 0
    return result


def promote_best_model(cleanup=True):
    result = eval_statistic() # should be the result of 1 latest model => clean up after statistic
    logger.info('###### CHECKING TOTAL WINRATE: %s', result)
    for model_name in result.keys():
        _, index = model_name.split('_')
        if result[model_name] > conf['EVALUATE_MARGIN']:
            logger.info('####### WE HAVE NEW BEST MODEL %s', model_name)
            # save best model as best.h5
            shutil.copyfile(os.path.join(conf['MODEL_DIR'], model_name + ".h5"), os.path.join(conf['MODEL_DIR'], conf['BEST_MODEL']))
            sync_model()  # copy best model to other self-play servers
            if cleanup:
                clean_up_result(result)
            return True
    logger.info("No new best model. Current result: %s", result)
    return False


def clean_up_result(result):
    for model_name in result.keys():
        shutil.rmtree(os.path.join(conf['EVAL_DIR'], model_name))

