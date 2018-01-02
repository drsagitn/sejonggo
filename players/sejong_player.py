from model import load_best_model
from self_play import new_tree, mcts_decision
import numpy.ma as ma
from play import legal_moves
from utils import str_coord


class SejongPlayer(object):
    def __init__(self, stop_exploration_move=0):
        self._model = load_best_model()
        self._mcts_tree = None
        self._stop_explore = stop_exploration_move
        self._current_move = 0

    def get_move(self, go_game):
        board = go_game.board
        policies, values = self._model.predict_on_batch(board)
        policy = policies[0]
        value = values[0]
        # resign = resign_model1 if current_model == model1 else resign_model2
        if False: # resign and value <= resign:
            return "resign"

        if not self._mcts_tree or not self._mcts_tree['subtree']:
            self._mcts_tree = new_tree(policy, board)

        mask = legal_moves(board)
        policy = ma.masked_array(policy, mask=mask)
        temperature = 1
        if self._current_move == self._stop_explore:
            temperature = 0
        index = mcts_decision(policy, board, None, self._mcts_tree, temperature, self._model)
        self._current_move += 1
        return index