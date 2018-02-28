import datetime
import numpy as np
from time import sleep
import numpy.ma as ma
import datetime

from conf import conf
from play import (
    index2coord, make_play, game_init,
    choose_first_player,
    show_board, get_winner, new_tree, top_one_with_virtual_loss, tree_depth
)
from predicting_queue_worker import put_predict_request

SIZE = conf['SIZE']
MCTS_BATCH_SIZE = conf['MCTS_BATCH_SIZE']
RESIGNATION_PERCENT = conf['RESIGNATION_PERCENT']
RESIGNATION_ALLOWED_ERROR = conf['RESIGNATION_ALLOWED_ERROR']
SELF_PLAY_DATA = conf['SELF_PLAY_DIR']


def update_root(result, node, action):
    from simulation_workers import lock
    lock.acquire()
    node['subtree'][action] = result
    node['count'] += 1
    node['value'] += result['value']
    node['mean_value'] = node['value'] / float(node['count'])
    result['virtual_loss'] = 0
    result['parent'] = node
    lock.release()

def error_handler(err):
    print(err)
    raise err


def async_simulate(node, board, model_indicator, energy, original_player):
    from simulation_workers import process_pool, basic_tasks
    from functools import partial
    results = []
    while energy > 0:
        action = top_one_with_virtual_loss(node)
        if action == {}:
            continue
        # print("ENERGY %s" % energy)
        child_node = action['node']
        child_node['parent'] = None
        child_node['virtual_loss'] += 2

        # callback_func = lambda result: update_root(result, node, action['action'])
        callback_func = partial(update_root, node=node, action=action['action'])
        # async_func = partial(basic_tasks, node=copy.deepcopy(child_node), board=board, move=action['action'], model=model, original_player=original_player)
        r = process_pool.apply_async(basic_tasks, (child_node, board, action['action'], model_indicator, original_player), callback=callback_func, error_callback=error_handler)
        results.append(r)
        # process_pool.apply_async(async_func, callback=callback_func)
        # result = basic_tasks(copy.deepcopy(child_node), np.copy(board), action['action'], model, original_player)
        # update_root(result, node, action['action'])
        energy -= 1
    # start = datetime.datetime.now()
    # while len(process_pool._cache):
    #     sleep(0.001)
    [result.wait() for result in results]
    # end = datetime.datetime.now()
    # print("###### WATING TIME %s", end - start)



def select_play(board, energy, mcts_tree, temperature, model_indicator):
    start = datetime.datetime.now()
    for i in range(int(conf['MCTS_SIMULATIONS']/conf['ENERGY'])):
        async_simulate(mcts_tree, np.copy(board), model_indicator, energy, board[0, 0, 0, -1])
    end = datetime.datetime.now()
    d = tree_depth(mcts_tree)
    print("################TIME PER MOVE: %s   tree depth: %s" % (end - start, d))
    if temperature == 1:
        total_n = sum(dic['count'] for dic in mcts_tree['subtree'].values())
        moves = []
        ps = []
        for move, dic in mcts_tree['subtree'].items():
            n = dic['count']
            if not n:
                continue
            p = dic['count'] / float(total_n)
            moves.append(move)
            ps.append(p)
        selected_a = np.random.choice(moves, size=1, p=ps)[0]
    elif temperature == 0:
        _, _, selected_a = max((dic['count'], dic['mean_value'], a) for a, dic in mcts_tree['subtree'].items())

    return selected_a

def play_game_async(model1_indicator, model2_indicator, energy, stop_exploration, self_play=False, num_moves=None, resign_model1=None, resign_model2=None):
    board, player = game_init()
    moves = []

    current_model_indicator, other_model_indicator = choose_first_player(model1_indicator, model2_indicator)
    mcts_tree, other_mcts = None, None

    last_value = None
    value = None

    model1_isblack = current_model_indicator == model1_indicator

    skipped_last = False
    temperature = 1
    start = datetime.datetime.now()
    end_reason = "PLAYED ALL MOVES"
    if num_moves is None:
        num_moves = SIZE * SIZE * 2

    for move_n in range(num_moves):
        last_value = value
        if move_n == stop_exploration:
            temperature = 0
        policy, value = put_predict_request(current_model_indicator, board, response_now=True)
        resign = resign_model1 if current_model_indicator == model1_indicator else resign_model2
        if resign and value <= resign:
            end_reason = "resign"
            break
        # Start of the game mcts_tree is None, but it can be {} if we selected a play that mcts never checked
        if not mcts_tree or not mcts_tree['subtree']:
            mcts_tree = new_tree(policy, board, add_noise=self_play)
            if self_play:
                other_mcts = mcts_tree

        index = select_play(board, energy, mcts_tree, temperature, current_model_indicator)
        x, y = index2coord(index)

        policy_target = np.zeros(SIZE*SIZE + 1)
        for _index, d in mcts_tree['subtree'].items():
            policy_target[_index] = d['p']

        move_data = {
            'board': np.copy(board),
            'policy': policy_target,
            'value': value,
            'move': (x, y),
            'move_n': move_n,
            'player': player ,
        }
        moves.append(move_data)

        if skipped_last and y == SIZE:
            end_reason = "BOTH_PASSED"
            break
        skipped_last = y == SIZE


        # Update trees
        if not self_play:
            # Update other only if we are not in self_play
            if other_mcts and index in other_mcts['subtree']:
                other_mcts = other_mcts['subtree'][index]
                other_mcts['parent'] = None # Cut the tree
        else:
            other_mcts = other_mcts['subtree'][index]
            other_mcts['parent'] = None # Cut the tree
        mcts_tree = mcts_tree['subtree'][index]
        mcts_tree['parent'] = None # Cut the tree

        # Swap players
        board, player = make_play(x, y, board)
        current_model_indicator, other_model_indicator = other_model_indicator, current_model_indicator
        mcts_tree, other_mcts = other_mcts, mcts_tree

        if conf['SHOW_EACH_MOVE']:
            # Inverted here because we already swapped players
            color = "W" if player == 1 else "B"

            print("%s(%s,%s)" % (color, x, y))
            print("")
            show_board(board)
            print("")


    winner, black_points, white_points = get_winner(board)
    player_string = {1: "B", 0: "D", -1: "W"}
    if end_reason == "resign":
        winner_string = "%s+R" % (player_string[player])
    else:
        winner_string = "%s+%s" % (player_string[winner], abs(black_points - white_points))
    winner_result = {1: 1, -1: 0, 0: None}

    if winner == 0:
        winner_model = None
    else:
        winner_model = model1_indicator if (winner == 1) == model1_isblack else model2_indicator

    if model1_isblack:
        modelB, modelW = model1_indicator, model2_indicator
    else:
        modelW, modelB = model1_indicator, model2_indicator

    if conf['SHOW_END_GAME']:
        if player == -1:
            # black played last
            bvalue, wvalue = value, last_value
        else:
            bvalue, wvalue = last_value, value
        print("")
        print("B:%s, W:%s" % (modelB.name, modelW.name))
        print("Bvalue:%s, Wvalue:%s" % (bvalue, wvalue))
        print(show_board(board))
        print("Game played (%s: %s) : %s" % (winner_string, end_reason, datetime.datetime.now() - start))

    game_data = {
        'moves': moves,
        'modelB_name': modelB.name,
        'modelW_name': modelW.name,
        'winner': winner_result[winner],
        'winner_model': winner_model.name,
        'result': winner_string,
        'resign_model1': resign_model1,
        'resign_model2': resign_model2,
    }
    return game_data
