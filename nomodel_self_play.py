import numpy as np
import datetime
from conf import conf
from play import (
    index2coord, make_play, game_init,
    choose_first_player,
    show_board, get_winner, new_tree, top_one_with_virtual_loss, tree_depth
)
from predicting_queue_worker import put_predict_request, put_name_request
from tree_util import find_best_leaf_virtual_loss, get_node_by_moves
import logging
from app_log import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

SIZE = conf['SIZE']
MCTS_BATCH_SIZE = conf['MCTS_BATCH_SIZE']
RESIGNATION_PERCENT = conf['RESIGNATION_PERCENT']
RESIGNATION_ALLOWED_ERROR = conf['RESIGNATION_ALLOWED_ERROR']
SELF_PLAY_DATA = conf['SELF_PLAY_DIR']
ENERGY = conf['ENERGY']


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
    print("Error in basic task", err)
    raise err


def back_propagation(result, node):
    leaf, moves = result
    closest_parent = get_node_by_moves(node, moves[:-1])
    leaf['virtual_loss'] = 0
    closest_parent['subtree'][moves[-1]]['parent'] = None  # cut old leaf node
    del closest_parent['subtree'][moves[-1]]['parent']  # also delete it
    closest_parent['subtree'][moves[-1]] = leaf  # attach new leaf node
    leaf['parent'] = closest_parent
    while True:
        closest_parent['count'] += 1
        closest_parent['value'] += leaf['value']
        closest_parent['mean_value'] = closest_parent['value'] / float(closest_parent['count'])
        closest_parent['virtual_loss'] = 0
        if closest_parent['parent']:
            closest_parent = closest_parent['parent']
        else:
            break


def async_simulate2(node, board, model_indicator, energy, original_player, process_id):
    if node['subtree'] == {}:
        return
    from simulation_workers import basic_tasks2, process_pool, simulation_result_queue
    pre_bp = 0
    while energy > 0:
        best_leaf, moves = find_best_leaf_virtual_loss(node)  # leaf is node with value = 0, count = 0, child = {}
        if best_leaf is not None and best_leaf['count'] > 0:  # already simulated leaf node
            energy -= 1
            pre_bp += 1
            continue
        if best_leaf is None:
            print("No best leaf at energy ", energy)
            r = simulation_result_queue[process_id].get()
            back_propagation(r, node)
            pre_bp += 1
            continue
        best_leaf['parent'] = None
        process_pool.apply_async(basic_tasks2, (best_leaf, board, moves, model_indicator, original_player, process_id),
                                 error_callback=error_handler)
        energy -= 1
    for i in range(ENERGY - pre_bp):
        r = simulation_result_queue[process_id].get()
        back_propagation(r, node)


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

def select_play(board, energy, mcts_tree, temperature, model_indicator, gpuid):
    start = datetime.datetime.now()
    for i in range(int(conf['MCTS_SIMULATIONS']/conf['ENERGY'])):
        async_simulate2(mcts_tree, np.copy(board), model_indicator, energy, board[0, 0, 0, -1], gpuid)
    end = datetime.datetime.now()
    try:
        d = tree_depth(mcts_tree)
        logger.debug("TIME PER MOVE: %s   tree depth: %s    1st level children: %s - %s" % (end - start, d, len(mcts_tree['subtree']), gpuid))
    except Exception as ex:
        logger.error(ex)

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

def play_game_async(model1_indicator, model2_indicator, energy, stop_exploration, process_id, self_play=False, num_moves=None, resign_model1=None, resign_model2=None):
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
        if conf['SHOW_EACH_MOVE'] and process_id == 0:
            pindex = [i for i, j in enumerate(policy) if j == max(policy)][0]  # get index of max policy
            x, y = index2coord(pindex)  # try to see where this policy advise to go
            print("%s to play max_p:%s  v:%s max_p_index(%s, %s)" % (board[0,0,0,-1], max(policy), value, x, y))
        resign = resign_model1 if current_model_indicator == model1_indicator else resign_model2
        if resign and value <= resign:
            end_reason = "resign"
            break
        # Start of the game mcts_tree is None, but it can be {} if we selected a play that mcts never checked
        if not mcts_tree or not mcts_tree['subtree']:
            mcts_tree = new_tree(policy, board, add_noise=self_play)
            if self_play:
                other_mcts = mcts_tree

        index = select_play(board, energy, mcts_tree, temperature, current_model_indicator, process_id)
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
            'player': player
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

        if conf['SHOW_EACH_MOVE'] and process_id == 0:
            # Inverted here because we already swapped players
            color = "W" if player == 1 else "B"
            print("%s(%s,%s) played by %s" % (color, x, y, other_model_indicator))
            print(show_board(board))


    winner, black_points, white_points = get_winner(board)
    player_string = {1: "B", 0: "D", -1: "W"}
    if end_reason == "resign":
        winner_string = "%s+R" % (player_string[player])
    else:
        winner_string = "%s+%s" % (player_string[winner], abs(black_points - white_points))
    winner_result = {1: 1, -1: 0, 0: None}

    if model1_isblack:
        modelB, modelW = model1_indicator, model2_indicator
    else:
        modelW, modelB = model1_indicator, model2_indicator

    modelB_name = put_name_request(modelB)
    modelW_name = put_name_request(modelW)

    if winner == 0:
        winner_model = None
    else:
        winner_model = modelB_name if (winner == 1) == model1_isblack else modelW_name

    if conf['SHOW_END_GAME']:
        if player == -1:
            # black played last
            bvalue, wvalue = value, last_value
        else:
            bvalue, wvalue = last_value, value
        print("")
        print("B:%s, W:%s" % (modelB_name, modelW_name))
        print("Bvalue:%s, Wvalue:%s" % (bvalue, wvalue))
        print("Resign threshold: %s" % resign)
        print(show_board(board))
        print("Game played (%s: %s) : %s" % (winner_string, end_reason, datetime.datetime.now() - start))

    game_data = {
        'moves': moves,
        'modelB_name': modelB_name,
        'modelW_name': modelW_name,
        'winner': winner_result[winner],
        'winner_model': winner_model,
        'result': winner_string,
        'resign_model1': resign_model1,
        'resign_model2': resign_model2,
    }
    return game_data
