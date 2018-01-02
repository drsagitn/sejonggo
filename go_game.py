from play import make_play, index2coord, game_init

WHITE = -1
BLACK = +1
EMPTY = 0
RESIGN = "resign"
PASS = "pass"

class GoGame(object):
    def __init__(self, size=9, komi=7.5):
        self.board, player = game_init()
        self.current_player = player
        self.size = size
        self.ko = None
        self.komi = komi
        self.handicaps = []
        self.history = []
        self.num_black_prisoners = 0
        self.num_white_prisoners = 0
        self.is_end_of_game = False
        # Each pass move by a player subtracts a point
        self.passes_white = 0
        self.passes_black = 0

    def do_move(self, action, color):
        if color is None:
            color = self.current_player
        if action != RESIGN and action != "pass":
            x, y = index2coord(action) # also included pass action. If skip action => y == SIZE
            make_play(x, y, self.board, color)


class IllegalMove(Exception):
    pass