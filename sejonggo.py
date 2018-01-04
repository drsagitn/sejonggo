import sys
from conf import conf
from play import game_init, index2coord, show_board, coord2index
from self_play import select_play, make_play, new_tree, show_tree
from model import load_best_model
import string
from __init__ import __version__
import numpy as np
import logging
from app_log import setup_logging
setup_logging()
logger = logging.getLogger("sejonggo")

COLOR_TO_PLAYER = {'B': 1, 'W': -1}
SIZE = conf['SIZE']


class SejongGoEngine(object):
    def __init__(self, model, mcts_simulations, board, resign=None, temperature=0, add_noise=False):
        self.model = model
        self.mcts_simulations = mcts_simulations
        self.resign = resign
        self.temperature = temperature
        self.board = board
        self.add_noise = add_noise
        self.mcts_tree = None
        self.move = 1

    def set_temperature(self, temperature):
        self.temperature = temperature

    def play(self, color, x, y, update_tree=True):
        index = coord2index(x, y)
        if update_tree:
            if self.mcts_tree and index in self.mcts_tree['subtree']:
                self.mcts_tree = self.mcts_tree['subtree'][index]
                self.mcts_tree['parent'] = None  # Cut the tree
            else:
                self.mcts_tree = None

        self.board, player = make_play(x, y, self.board, color)
        self.move += 1
        return self.board, player

    def genmove(self, color):
        policies, values = self.model.predict_on_batch(self.board)
        policy = policies[0]
        value = values[0]
        if self.resign and value <= self.resign:
            x = 0
            y = SIZE + 1
            return x, y, policy, value, self.board, self.player

        if not self.mcts_tree or not self.mcts_tree['subtree']:
            self.mcts_tree = new_tree(policy, self.board)

        index = select_play(policy, self.board, self.mcts_simulations, self.mcts_tree, self.temperature, self.model)
        logger.info("Generated index %s", index)
        x, y = index2coord(index)
        # show_tree(x, y, self.mcts_tree)

        policy_target = np.zeros(SIZE*SIZE + 1)
        for _index, d in self.mcts_tree['subtree'].items():
            policy_target[_index] = d['p']

        self.board, self.player = self.play(color, x, y)
        return x, y, policy_target, value, self.board, self.player

class GTPEngine(object):
    def __init__(self):
        self._komi = 0
        self.board, self.player = game_init()
        model = load_best_model()
        self.sejong_engine = SejongGoEngine(model, conf['MCTS_SIMULATIONS'], self.board)
        print("GTP engine ready")

    def name(self):
        return "SejongGo - {} - {} simulations".format(self.sejong_engine.model.name, conf['MCTS_SIMULATIONS'])

    def version(self):
        return __version__

    def protocol_version(self):
        return "2"

    def list_commands(self):
        return ""

    def boardsize(self, size):
        return ""

    def komi(self, komi):
        self._komi = komi
        return ""

    def parse_move(self, move):

        if move == 'pass':
            x, y = 0, SIZE
        else:
            letter = move[0]
            number = move[1:]

            x = string.ascii_uppercase.index(letter)
            if x >= 9:
                x -= 1 # I is a skipped letter
            y = int(number) - 1

        x, y = x, SIZE - y - 1
        return x, y

    def print_move(self, x, y):
        x, y = x, SIZE - y - 1

        if x >= 8:
            x += 1 # I is a skipped letter

        move = string.ascii_uppercase[x] + str(y + 1)
        return move

    def play(self, color, move):
        announced_player = COLOR_TO_PLAYER[color]
        assert announced_player == self.player
        x, y = self.parse_move(move)
        self.board, self.player = self.sejong_engine.play(announced_player, x, y)
        return ""

    def genmove(self, color):
        announced_player = COLOR_TO_PLAYER[color]
        # assert announced_player == self.player
        # genmove, play and return update board, player
        x, y, policy_target, value, self.board, self.player = self.sejong_engine.genmove(announced_player)

        move_string = self.print_move(x, y)
        result = move_string
        return result

    def clear_board(self):
        self.board, self.player = game_init()
        return ""

    def parse_command(self, line):
        tokens = line.strip().split(" ")
        command = tokens[0]
        args = tokens[1:]
        # try:
        method = getattr(self, command)
        result = method(*args)
        if not result.strip():
            return "=\n\n"
        return "= " + result + "\n\n"
        # except AttributeError:
        #     return "= Command not found" + "\n\n"


def main():
    engine = GTPEngine()
    inpt_fn = input
    while True:
        inpt = inpt_fn()
        cmd_list = inpt.split("\n")
        for cmd in cmd_list:
            logger.info("<<< " + cmd)
            result = engine.parse_command(cmd)
            if result.strip():
                sys.stdout.write(result)
                sys.stdout.flush()
                logger.info("Next turn player:" + str(engine.player))
                logger.info(">>> " + result)
                # print(show_board(engine.board))



if __name__ == "__main__":
    main()
