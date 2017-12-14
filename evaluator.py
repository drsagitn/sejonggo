from self_play import play_game, self_play, save_game_data
from conf import conf
import os
from tqdm import tqdm
import datetime
from model import load_best_model, load_model
import shutil
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
                total += 1
                if os.path.isfile(os.path.join(model_dir, game_dir, model_name)):
                    wins += 1
        result[model_name] = wins/total if total != 0 else 0
    return result


def promote_best_model(cleanup=True):
    result = eval_statistic()
    logger.info('Evaluation result: %s', result)
    best_model = load_best_model()
    _, best_index = best_model.name.split('_')
    for model_name in result.keys():
        _, index = model_name.split('_')
        if index > best_index and result[model_name] > conf['EVALUATE_MARGIN']:
            save_as_best_model(model_name)
            logger.info('We have new best model %s', model_name)
            if cleanup:
                clean_up_result(result)
            return
    logger.info("No new best model. Current result: %s", result)


def save_as_best_model(model_name):
    full_filename = os.path.join(conf['MODEL_DIR'], conf['BEST_MODEL'])
    model = load_model(conf['MODEL_DIR'], model_name)
    model.save(full_filename)


def clean_up_result(result):
    for model_name in result.keys():
        shutil.rmtree(os.path.join(conf['EVAL_DIR'], model_name))

