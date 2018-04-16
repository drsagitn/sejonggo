from sgfmill import sgf
from conf import conf
from play import get_real_board
import os
import h5py
import numpy as np
import shutil
import logging
from app_log import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

SIZE = conf['SIZE']
SELF_PLAY_DATA = conf['SELF_PLAY_DIR']

def save_file(model_name, game_n, move_data, winner):
    board = move_data['board']
    policy_target = move_data['policy']
    player = move_data['player']
    value_target = 1 if winner == player else -1
    move = move_data['move_n']
    directory = os.path.join(conf["GAMES_DIR"], model_name, "game_%03d" % game_n, "move_%03d" % move)
    try:
        os.makedirs(directory)
    except OSError:
        while True:
            game_n += 1
            directory = os.path.join(conf['GAMES_DIR'], model_name, "game_%03d" % game_n, "move_%03d" % move)
            try:
                os.makedirs(directory)
                break
            except OSError:
                pass

    with h5py.File(os.path.join(directory, 'sample.h5'),'w') as f:
        f.create_dataset('board', data=board, dtype=np.float32)
        f.create_dataset('policy_target', data=policy_target, dtype=np.float32)
        f.create_dataset('value_target', data=np.array(value_target), dtype=np.float32)


def save_game_data(model_name, game_n, game_data):
    winner = game_data['winner']
    for move_data in game_data['moves']:
        save_file(model_name, game_n, move_data, winner)
    if conf['SGF_ENABLED']:
        save_game_sgf(model_name, game_n, game_data)


def save_self_play_data(model_name, game_no, game_data):
    winner = game_data['winner']
    logger.debug("Saving self-play game %s, num move %s, result %s", game_no, len(game_data['moves']), game_data['result'])
    for move_data in game_data['moves']:
        board = move_data['board']
        policy_target = move_data['policy']
        player = move_data['player']
        value_target = 1 if winner == player else -1
        move = move_data['move_n']
        directory = os.path.join(SELF_PLAY_DATA, model_name, "game_%05d" % game_no, "move_%03d" % move)
        try:
            os.makedirs(directory)
        except OSError:
            while True:
                game_no += 1
                directory = os.path.join(SELF_PLAY_DATA, model_name, "game_%05d" % game_no, "move_%03d" % move)
                try:
                    os.makedirs(directory)
                    break
                except OSError:
                    pass
                except Exception as e:
                    logger.exception("fail to make dir2")
        except Exception as e:
            logger.exception("fail to make dir")
        with h5py.File(os.path.join(directory, 'sample.h5'), 'w') as f:
            f.create_dataset('board', data=board, dtype=np.float32)
            f.create_dataset('policy_target', data=policy_target, dtype=np.float32)
            f.create_dataset('value_target', data=np.array(value_target), dtype=np.float32)
    if conf['SGF_ENABLED']:
        save_game_sgf(model_name, game_no, game_data)


# In this dir, remove all self-play games with num of move < 50
def clean_up(self_play_dir, min_move):
    total = 0;
    for model_dir in os.listdir(self_play_dir):
        for game_dir in os.listdir(os.path.join(self_play_dir, model_dir)):
            real_path = os.path.join(self_play_dir, model_dir, game_dir)
            try:
                num_move = len(os.listdir(real_path))
                if num_move < min_move:
                    shutil.rmtree(real_path)
                    total += 1
                    logger.info("Remove %s, number of moves is %s", real_path, num_move)
            except OSError:
                pass
    logger.info("Total removed %s", total)


def statistic_all_model(self_play_dir,min_move):
    stat = {}
    for i in range(min_move):
        stat[i] = []
    for model_dir in os.listdir(self_play_dir):
        for game_dir in os.listdir(os.path.join(self_play_dir, model_dir)):
            real_path = os.path.join(self_play_dir, model_dir, game_dir)
            try:
                num_move = len(os.listdir(real_path))
                if num_move < min_move:
                    stat[num_move].append(game_dir)
            except OSError:
                pass
    return stat


def statistic_by_model(model_self_play_dir,min_move):
    stat = {}
    for i in range(min_move):
        stat[i] = []

    for game_dir in os.listdir(model_self_play_dir):
        real_path = os.path.join(model_self_play_dir, game_dir)
        try:
            num_move = len(os.listdir(real_path))
            if num_move < min_move:
                stat[num_move].append(game_dir)
        except OSError:
            pass
    return stat

def save_game_sgf(model_name, game_n, game_data):
    game = sgf.Sgf_game(size=SIZE)

    modelB_name = game_data['modelB_name']
    modelW_name = game_data['modelW_name']
    result = game_data['result']

    game.root.set_raw("PB", modelB_name.encode('utf-8'))
    game.root.set_raw("PW", modelW_name.encode('utf-8'))
    game.root.set_raw("KM", str(conf['KOMI']).encode('utf-8'))

    game.root.set_raw("RE", result.encode('utf-8'))
    for move_data in game_data['moves']:
        node = game.extend_main_sequence()
        color = 'b' if move_data['player'] == 1 else 'w'
        x, y = move_data['move']

        move = (SIZE - 1 - y, x) if y != SIZE else None  # Different orienation

        move_n = move_data['move_n']
        next_board = game_data['moves'][(move_n + 1) % len(game_data['moves'])]['board']
        comment = "Value %s\n %s" % (move_data['value'], get_real_board(next_board))
        node.set("C", comment)
        node.set_move(color, move)


    try:
        os.makedirs(os.path.join(conf['GAMES_DIR'], model_name))
    except OSError:
        pass

    filename = os.path.join(conf["GAMES_DIR"], model_name, "game_%03d.sgf" % game_n)
    while os.path.isfile(filename):
        game_n += 1
        filename = os.path.join(conf["GAMES_DIR"], model_name, "game_%03d.sgf" % game_n)

    with open(filename, "wb") as f:
        f.write(game.serialise())
