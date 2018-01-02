import sys
import gtp
from go_game import GoGame, IllegalMove
from play import str_coord, gtpcoord2index
from go_game import BLACK, WHITE, PASS, RESIGN

def parse_color(color):
    if color.lower() in ["b", "black"]:
        return BLACK
    elif color.lower() in ["w", "white"]:
        return WHITE
    else:
        return False


def parse_vertex(vertex_string):
    if vertex_string is None:
        return False
    elif vertex_string.lower() == "pass":
        return PASS
    elif len(vertex_string) > 1:
        x = "abcdefghjklmnopqrstuvwxyz".find(vertex_string[0].lower()) + 1
        if x == 0:
            return False
        if vertex_string[1:].isdigit():
            y = int(vertex_string[1:])
        else:
            return False
    else:
        return False
    return (x, y)


def parse_move(move_string):
    color_string, vertex_string = (move_string.split(" ") + [None])[:2]
    color = parse_color(color_string)
    if color is False:
        return False
    vertex = parse_vertex(vertex_string)
    if vertex is False:
        return False

    return color, vertex


class ExtendedGtpEngine(gtp.Engine):
    def cmd_genmove(self, arguments):
        c = parse_color(arguments)
        if c:
            move = self._game.get_move(c)
            self._game.make_move(c, move)
            return str_coord(move)
        else:
            raise ValueError("unknown player: {}".format(arguments))

    def cmd_play(self, arguments):
        move = parse_move(arguments)
        if move:
            color, vertex = move
            if self.vertex_in_range(vertex):
                if vertex == RESIGN or vertex == PASS:
                    action = vertex
                else:
                    x, y = vertex
                    action = gtpcoord2index(x, y)
                if self._game.make_move(color, action):
                    return
        raise ValueError("illegal move")

class GTPGameConnector(object):
    def __init__(self, player):
        self._state = GoGame()
        self._player = player

    def clear(self):
        self._state = GoGame(self._state.size)

    def get_move(self, color):
        self._state.current_player = color
        move = self._player.get_move(self._state)
        return move

    def make_move(self, color, vertex):
        try:
            self._state.do_move(vertex, color)
            return True
        except IllegalMove:
            return False

    def set_size(self, n):
        self._state = GoGame(n)

    def set_komi(self, k):
        self._state.komi = k

def run_gtp(player_obj, inpt_fn=None, name="Sejong Go Gtp Player", version="0.1"):
    gtp_game = GTPGameConnector(player_obj)
    gtp_engine = ExtendedGtpEngine(gtp_game, name, version)
    if inpt_fn is None:
        inpt_fn = input

    sys.stderr.write("Sejong GTP engine ready\n")
    sys.stderr.flush()
    while not gtp_engine.disconnect:
        inpt = inpt_fn()
        # handle either single lines at a time
        # or multiple commands separated by '\n'
        cmd_list = inpt.split("\n")
        for cmd in cmd_list:
            engine_reply = gtp_engine.send(cmd)
            sys.stdout.write(engine_reply)
            sys.stdout.flush()