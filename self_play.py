# -*- coding: utf-8 -*-
import datetime
import numpy as np
import tqdm
import os
import numpy.ma as ma
from sgfsave import save_self_play_data, save_game_data
from play import (
    legal_moves, index2coord, make_play, game_init,
    choose_first_player,
    show_board, get_winner, new_tree, top_n_actions, new_subtree, top_one_action, tree_depth
)
from symmetry import random_symmetry_predict
from random import random
from conf import conf

import logging
from app_log import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

SIZE = conf['SIZE']
MCTS_BATCH_SIZE = conf['MCTS_BATCH_SIZE']
RESIGNATION_PERCENT = conf['RESIGNATION_PERCENT']
RESIGNATION_ALLOWED_ERROR = conf['RESIGNATION_ALLOWED_ERROR']
SELF_PLAY_DATA = conf['SELF_PLAY_DIR']

def simulate(node, board, model, mcts_batch_size, original_player):
    node_subtree = node['subtree']
    max_actions = top_n_actions(node_subtree, mcts_batch_size)
    selected_action = max_actions[0]['action']
    selected_node = node_subtree[selected_action]
    if selected_node['subtree'] == {}:

        if False: #conf['THREAD_SIMULATION']:
            from simulation_workers import process_pool, board_worker
            ret = process_pool.map(board_worker, [(dic, board) for i, dic in enumerate(max_actions)])
            boards = np.array(ret)
        else:
            boards = np.zeros((len(max_actions), SIZE, SIZE, 17), dtype=np.float32)
            for i, dic in enumerate(max_actions):
                action = dic['action']
                tmp_board = np.copy(board)

                if dic['node']['subtree'] != {}:
                    # already expanded
                    tmp_node = dic['node']
                    tmp_action = action
                    x, y = index2coord(tmp_action)
                    tmp_board, _ = make_play(x, y, tmp_board)
                    while tmp_node['subtree'] != {}:
                        # tmp_max_actions = top_n_actions(tmp_node['subtree'], 1)
                        # tmp_d = tmp_max_actions[0]
                        tmp_d = top_one_action(tmp_node['subtree'])
                        tmp_node = tmp_d['node']
                        tmp_action = tmp_d['action']
                        # The node for this action is the leaf, this is where the
                        # update will start, working up the tree
                        dic['node'] = tmp_node
                        x, y = index2coord(tmp_action)
                        make_play(x, y, tmp_board)
                    boards[i] = tmp_board
                else:
                    x, y = index2coord(action)
                    make_play(x, y, tmp_board)
                    boards[i] = tmp_board

        # The random symmetry will changes boards, so copy them before hand
        presymmetry_boards = np.copy(boards)
        policies, values = random_symmetry_predict(model, boards)

        if conf['THREAD_SIMULATION']:
            from simulation_workers import subtree_worker, process_pool
            subtree_array = process_pool.map(subtree_worker, [(policy, board) for policy, board in zip(policies, presymmetry_boards)])

            for subtree, board, v, action in zip(subtree_array, presymmetry_boards, values, max_actions):
                player = board[0, 0, -1]
                value = v[0] if player == original_player else -v[0]
                leaf_node = action['node']
                for _, node in subtree.items():
                    node['parent'] = leaf_node
                leaf_node['subtree'] = subtree

                current_node = leaf_node
                while True:
                    current_node['count'] += 1
                    current_node['value'] += value
                    current_node['mean_value'] = current_node['value'] / float(current_node['count'])
                    if current_node['parent']:
                        current_node = current_node['parent']
                    else:
                        break

        else:
            for policy, v, board, action in zip(policies, values, presymmetry_boards, max_actions):
                # reshape from [n, n, 17] to [1, n, n, 17]
                shape = board.shape
                board = board.reshape([1] + list(shape))

                player = board[0, 0, 0, -1]
                # Inverse value if we're looking from other player perspective
                value = v[0] if player == original_player else -v[0]

                leaf_node = action['node']
                subtree = new_subtree(policy, board, leaf_node)
                leaf_node['subtree'] = subtree

                current_node = leaf_node
                while True:
                    current_node['count'] += 1
                    current_node['value'] += value
                    current_node['mean_value'] = current_node['value'] / float(current_node['count'])
                    if current_node['parent']:
                        current_node = current_node['parent']
                    else:
                        break
    else:
        x, y = index2coord(selected_action)
        make_play(x, y, board)
        simulate(selected_node, board, model, mcts_batch_size, original_player)


def mcts_decision(policy, board, mcts_simulations, mcts_tree, temperature, model):
    # TODO: make parallelization here, each simulation can be handled by a thread/process/CPU
    if mcts_simulations is None:
        mcts_simulations = conf['MCTS_SIMULATIONS']

    for i in range(int(mcts_simulations/MCTS_BATCH_SIZE)):  # depth of the tree
        test_board = np.copy(board)
        original_player = board[0,0,0,-1]
        start = datetime.datetime.now()
        simulate(mcts_tree, test_board, model, MCTS_BATCH_SIZE, original_player)
        end = datetime.datetime.now()

    # from play import show_tree
    # show_tree(None, None, mcts_tree)

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

def select_play(policy, board, mcts_simulations, mcts_tree, temperature, model):
    mask = legal_moves(board)
    policy = ma.masked_array(policy, mask=mask)
    start = datetime.datetime.now()
    index = mcts_decision(policy, board, mcts_simulations, mcts_tree, temperature, model)
    end = datetime.datetime.now()
    d = tree_depth(mcts_tree)
    # print("################TIME PER MOVE: %s   tree depth: %s" % (end - start, d))
    return index

def play_game(model1, model2, mcts_simulations, stop_exploration, self_play=False, num_moves=None, resign_model1=None, resign_model2=None):
    board, player = game_init()
    moves = []

    current_model, other_model = choose_first_player(model1, model2)
    mcts_tree, other_mcts = None, None

    last_value = None
    value = None

    model1_isblack = current_model == model1

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
        policies, values = current_model.predict_on_batch(board)
        policy = policies[0]
        value = values[0]
        resign = resign_model1 if current_model == model1 else resign_model2
        if resign and value <= resign:
            end_reason = "resign"
            break
        # Start of the game mcts_tree is None, but it can be {} if we selected a play that mcts never checked
        if not mcts_tree or not mcts_tree['subtree']:
            mcts_tree = new_tree(policy, board, add_noise=self_play)
            if self_play:
                other_mcts = mcts_tree

        index = select_play(policy, board, mcts_simulations, mcts_tree, temperature, current_model)
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
        current_model, other_model = other_model, current_model
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
        winner_model = model1 if (winner == 1) == model1_isblack else model2

    if model1_isblack:
        modelB, modelW = model1, model2
    else:
        modelW, modelB = model1, model2

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


def model_self_play(model, one_game_only=-1):
    n_games = conf['N_GAMES']
    mcts_simulations = conf['MCTS_SIMULATIONS']

    desc = "Self play %s" % model.name
    games = tqdm.tqdm(range(n_games), desc=desc)
    games_data = []
    current_resign = None
    min_values = []
    for game in games:
        if 0 <= one_game_only and game != one_game_only:
            continue
        directory = os.path.join(SELF_PLAY_DATA, model.name, "game_%05d" % game)
        if os.path.isdir(directory):
            continue
        os.makedirs(directory)

        if random() > RESIGNATION_PERCENT:
            resign = current_resign
        else:
            resign = None

        start = datetime.datetime.now()
        game_data = play_game(model, model, mcts_simulations, conf['STOP_EXPLORATION'], self_play=True, resign_model1=resign, resign_model2=resign)
        stop = datetime.datetime.now()

        # If we did not use resignation, we had the result towards resign value.
        if resign == None:
            winner = game_data['winner']
            if winner == 1:
                min_value = min([move['value'] for move in game_data['moves'][::2]])
            else:
                min_value = min([move['value'] for move in game_data['moves'][1::2]])
            min_values.append(min_value)
            l = len(min_values)
            resignation_index = int(RESIGNATION_ALLOWED_ERROR * l)
            if resignation_index > 0:
                current_resign = min_values[resignation_index]

        moves = len(game_data['moves'])
        speed = ((stop - start).seconds / moves) if moves else 0.
        games.set_description(desc + " %s moves %.2fs/move " % (moves, speed))
        save_self_play_data(model.name, game, game_data)
        logger.info("Finish self-play game %s", game)
        games_data.append(game_data)
        if one_game_only >= 0:
            break
    return games_data


def self_play(model, n_games, mcts_simulations):
    desc = "Self play %s" % model.name
    games = tqdm.tqdm(range(n_games), desc=desc)
    games_data = []
    current_resign = None
    min_values = []
    for game in games:

        if random() > RESIGNATION_PERCENT:
            resign = current_resign
        else:
            resign = None

        start = datetime.datetime.now()
        game_data = play_game(model, model, mcts_simulations, conf['STOP_EXPLORATION'], self_play=True, resign_model1=resign, resign_model2=resign)
        stop = datetime.datetime.now()

        # If we did not use resignation, we had the result towards resign value.
        if resign == None:
            winner = game_data['winner']
            if winner == 1:
                min_value = min([move['value'] for move in game_data['moves'][::2]])
            else:
                min_value = min([move['value'] for move in game_data['moves'][1::2]])
            min_values.append(min_value)
            l = len(min_values)
            resignation_index = int(RESIGNATION_ALLOWED_ERROR * l)
            if resignation_index > 0:
                current_resign = min_values[resignation_index]

        moves = len(game_data['moves'])
        speed = ((stop - start).seconds / moves) if moves else 0.
        games.set_description(desc + " %s moves %.2fs/move " % (moves, speed))
        save_game_data(model.name, game, game_data)
        games_data.append(game_data)
    return games_data

