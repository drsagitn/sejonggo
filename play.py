from random import random
import numpy as np
from math import sqrt
from numpy.ma.core import MaskedConstant
import numpy.ma as ma
from conf import conf
import logging
from app_log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
###### board and game util functions

SIZE = conf['SIZE']
SWAP_INDEX = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]
colstr = 'ABCDEFGHJKLMNOPQRST'
W = SIZE + 2
Cpuct = 1
DIRICHLET_ALPHA = conf['DIRICHLET_ALPHA']
DIRICHLET_EPSILON = conf['DIRICHLET_EPSILON']

def str_coord(c):
    if c == "resign":
        return c
    if SIZE*SIZE == c:
        return 'pass'
    row, col = divmod(c - (W+1), W)
    return '%c%d' % (colstr[col], SIZE - row)


def index2coord(index):
    y = index // SIZE
    x = index - SIZE * y
    return x, y

def coord2index(x, y):
    return y * SIZE + x

def gtpcoord2index(x,y):
    x -= 1
    y -= 1
    index = SIZE * y + x
    return index

def get_surrounding(x,y):
    ret_array = []
    if y - 1 >= 0:
        ret_array.append((x, y-1))
    if x + 1 < SIZE:
        ret_array.append((x + 1, y))
    if y + 1 < SIZE:
        ret_array.append((x, y + 1))
    if x - 1 >= 0:
        ret_array.append((x-1, y))
    return ret_array

def get_liberties(x, y, board, color=None, parent_x=None, parent_y=None):
    ret_array = []
    real_board = get_real_board(board)
    if color is None:
        color = board[0,0,0,-1]
    for (a,b) in get_surrounding(x, y):
        if a == parent_x and b == parent_y: # skip previous node
            continue
        if real_board[b,a] == 0:
            ret_array.append((a,b))
        elif real_board[b,a] == color:
            ret_array = list(set(get_liberties(a, b, board, color, x, y) + ret_array))
    return ret_array

def legal_moves(board):
    # Occupied places
    mask1 = board[0,:,:,0].reshape(-1) != 0
    mask2 = board[0,:,:,1].reshape(-1) != 0
    mask = mask1 + mask2

    # Ko situations
    ko_mask = ((board[0,:,:,2] - board[0,:,:,0]))
    if (ko_mask == 1).sum() == 1:
        mask += (ko_mask == 1).reshape(-1)

    #suicide situation
    index = -1
    player = board[0, 0, 0, -1]
    real_board = get_real_board(board)
    for move in mask:
        index += 1
        if move == 0:
            col, row = index2coord(index)
            copy_board = np.copy(real_board)
            copy_board[row, col] = player
            capture_others = False
            for (rs, cs) in get_surrounding(row, col):
                if player != real_board[rs, cs] and capture_group(cs, rs, copy_board):
                    capture_others = True
                    break
            if capture_others:
                continue  # if this move capture any surrounding stones, allow it even it is suicide move
            if capture_group(col, row, real_board):
                mask[index] = 1

    # Pass is always legal
    mask = np.append(mask, 0)
    return mask

def get_real_board(board):
    player = board[0,0,0,-1]
    if player == 1:
        real_board = board[0,:,:,0] - board[0,:,:,1]
    else:
        real_board = board[0,:,:,1] - board[0,:,:,0]
    return real_board

def _show_board(board, policy):
    real_board = get_real_board(board)
    if policy is not None:
        index = policy.argmax()
        x, y = index2coord(index)
    string = ""
    for j, row in enumerate(real_board):
        for i, c in enumerate(row):
            if c == 1:
                string += u"○ "
            elif c == -1:
                string += u"● "
            elif policy is not None and i == x and j == y:
                string += u"X "
            else:
                string += u". "
        string += "\n"
    if policy is not None and y == SIZE:
        string += "Pass policy"
    return string


def show_board(board, policy=None, history=1):
    results = []
    for i in reversed(range(history)):
        tmp_board = np.copy(board)
        tmp_board = tmp_board[:,:,:,i:]
        if i % 2 == 1:
            tmp_board[:,:,:,-1] *= -1
        results.append(_show_board(tmp_board, policy))
    return "\n".join(results)


def show_board_old(board):
    real_board = get_real_board(board)
    for row in real_board:
        for c in row:
            if c == 1:
                print(u"○", end=' ')
            elif c == -1:
                print(u"●", end=' ')
            else:
                print(u".", end=' ')
        print("")

dxdys = [(1, 0), (-1, 0), (0, 1), (0, -1)]
def capture_group(x, y, real_board, group=None):
    if group is None:
        group = [(x, y)]

    c = real_board[y][x]
    for dx, dy in dxdys:
        nx = x + dx
        ny = y + dy
        if (nx, ny) in group:
            continue
        if not(0 <= nx < SIZE and 0 <= ny < SIZE):
            continue
        dc = real_board[ny][nx]
        if dc == 0:
            return None
        elif dc == c:
            group.append( (nx, ny) )
            group = capture_group(nx, ny, real_board, group=group)
            if group == None:
                return None
    return group

def take_stones(x, y, board):
    real_board = get_real_board(board)
    _player = 1 if board[0,0,0,-1] == 1 else -1
    for dx, dy in dxdys:  # We need to check capture
        nx = x + dx
        ny = y + dy
        if not(0 <= nx < SIZE and 0 <= ny < SIZE):
            continue
        if real_board[ny][nx] == 0:
            continue
        if real_board[ny][nx] == _player:
            continue
        group = capture_group(nx, ny, real_board)
        if group:
            for _x, _y in group:
                assert board[0,_y,_x,1] == 1
                board[0,_y,_x,1] = 0
                real_board[_y][_x] = 0
    for dx, dy in dxdys + [(0, 0)]:  # We need to check self sucide.
        nx = x + dx
        ny = y + dy
        if not(0 <= nx < SIZE and 0 <= ny < SIZE):
            continue
        if real_board[ny][nx] == 0:
            continue
        if real_board[ny][nx] != _player:
            continue
        group = capture_group(nx, ny, real_board)
        if group:
            for _x, _y in group:
                # Sucide
                assert board[0,_y,_x,0] == 1
                board[0,_y,_x,0] = 0
                real_board[_y][_x] = 0

    return board

def swap_player(board):
    player = board[0,0,0,-1]
    board[:, :, :, range(16)] = board[:, :, :, SWAP_INDEX]
    player = -1 if player == 1 else 1
    board[:, :, :, -1] = player
    return player

def make_play(x, y, board, color=None):
    if color is not None and color != board[0,0,0,-1]:
        player = swap_player(board)
    else:
        player = board[0,0,0,-1]
    board[:,:,:,2:16] = board[:,:,:,0:14]
    if y != SIZE:
        assert board[0,y,x,1] == 0
        assert board[0,y,x,0] == 0
        board[0,y,x,0] = 1  # Careful here about indices
        board = take_stones(x, y, board)
    else:
        # "Skipping", player
        pass
    # swap_players
    swap_player(board)
    return board, player

def _color_adjoint(i, j, color, board):
    # TOP
    SIZE1 = len(board)
    SIZE2 = len(board[0])
    if i > 0 and board[i-1][j] == 0:
        board[i-1][j] = color
        _color_adjoint(i - 1, j, color, board)
    # BOTTOM
    if i < SIZE1 - 1 and board[i+1][j] == 0:
        board[i+1][j] = color
        _color_adjoint(i + 1, j, color, board)
    # LEFT
    if j > 0 and board[i][j - 1] == 0:
        board[i][j - 1] = color
        _color_adjoint(i, j - 1, color, board)
    # RIGHT
    if j < SIZE2 - 1 and board[i][j + 1] == 0:
        board[i][j + 1] = color
        _color_adjoint(i, j + 1, color, board)
    return board

def color_board(real_board, color):
    board = np.copy(real_board)
    for i, row in enumerate(board):
        for j, v in enumerate(row):
            if v == color:
                _color_adjoint(i, j, color, board)
    return board


def get_winner(board):
    real_board = get_real_board(board)
    points =  _get_points(real_board)
    black = points.get(1, 0) + points.get(2, 0)
    white = points.get(-1, 0) + points.get(-2, 0) + conf['KOMI']
    if black > white:
        return 1, black, white
    elif black == white:
        return 0, black, white
    else:
        return -1, black, white

def _get_points(real_board):
    colored1 = color_board(real_board,  1)
    colored2 = color_board(real_board, -1)
    total = colored1 + colored2
    unique, counts = np.unique(total, return_counts=True)
    points = dict(zip(unique, counts))
    return points


def game_init():
    board = np.zeros((1, SIZE, SIZE, 17), dtype=np.int32)
    player = 1
    board[:,:,:,-1] = player
    return board, player

def choose_first_player(model1, model2):
    if random() < .5:
        current_model, other_model = model1, model2
    else:
        other_model, current_model = model1, model2
    return current_model, other_model

def top_one_with_virtual_loss(node):
    subtree = node['subtree']
    total_n = sqrt(sum(dic['count'] for dic in subtree.values()))
    if total_n == 0:
        total_n = 1
    max_value = -100
    max_action = {}
    for a, dic in subtree.items():
        if dic['virtual_loss'] > 0:
            continue
        u = Cpuct * dic['p'] * total_n / (1. + dic['count'])
        v = dic['mean_value'] + u
        if v > max_value:
            max_value = v
            max_action = {'action': a, 'node': dic}
    return max_action

def top_one_action(subtree):
    total_n = sqrt(sum(dic['count'] for dic in subtree.values()))
    if total_n == 0:
        total_n = 1
    max_action = {'action': -1, 'value': -1, 'node': None}
    for a, dic in subtree.items():
        u = Cpuct * dic['p'] * total_n / (1. + dic['count'])
        v = dic['mean_value'] + u
        if v > max_action['value']:
            max_action = {'action': a, 'value': v, 'node': dic}
    return max_action

def top_n_actions(subtree, top_n):
    total_n = sqrt(sum(dic['count'] for dic in subtree.values()))
    if total_n == 0:
        total_n = 1
    # Select exploration
    max_actions = []
    for a, dic in subtree.items():
        u = Cpuct * dic['p'] * total_n / (1. + dic['count'])
        v = dic['mean_value'] + u

        if len(max_actions) < top_n or v > max_actions[-1]['value']:
            max_actions.append({'action': a, 'value': v, 'node': dic})
            max_actions.sort(key=lambda x: x['value'], reverse=True)
        if len(max_actions) > top_n:
            max_actions = max_actions[:-1]
    return max_actions


def tree_depth(tree):
    if tree['subtree'] is None:
        return 1
    md = 0
    for key, node in tree['subtree'].items():
        d = tree_depth(node)
        if d > md:
            md = d
    return md + 1



def show_tree(x, y, tree, indent=''):
    if tree['parent'] is None:
        print('ROOT p: %s, count: %s' % (tree['p'], tree['count']))
    elif tree['count'] >= 1:
        print('%s Move(%s,%s) p: %s, count: %s' % (indent, x, y, tree['p'], tree['count']))
    for action, node in tree['subtree'].items():
        x, y = index2coord(action)
        show_tree(x, y, node, indent=indent+'--')

def new_tree(policy, board, add_noise=False):
    mcts_tree = {
        'count': 0,
        'value': 0,
        'mean_value': 0,
        'p': 1,
        'subtree': {},
        'parent': None,
        'virtual_loss': 0
    }
    subtree = new_subtree(policy, board, parent=mcts_tree, add_noise=add_noise)
    mcts_tree['subtree'] = subtree
    return mcts_tree

def new_subtree(policy, board, parent, add_noise=False):
    leaf = {}

    # We need to check for legal moves here because MCTS might not have expanded
    # this subtree
    mask = legal_moves(board)
    policy = ma.masked_array(policy, mask=mask)

    # Add Dirichlet noise.
    tmp = policy.reshape(-1)
    if add_noise:
        noise = np.random.dirichlet([DIRICHLET_ALPHA for i in range(tmp.shape[0])])
        tmp = (1 - DIRICHLET_EPSILON) * tmp + DIRICHLET_EPSILON * noise


    for move, p in enumerate(tmp):
        if isinstance(p, MaskedConstant):
            continue

        leaf[move] = {
            'count': 0,
            'value': 0,
            'mean_value': 0,
            'p': p,
            'subtree': {},
            'parent': parent,
            'virtual_loss': 0
        }
    if len(leaf) == 1: # if there is only one possible move left (move of skip) the return subtree = {} to stop simulation
        return {}
    return leaf
