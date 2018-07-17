from multiprocessing import Process
import datetime
import tqdm
import os
import numpy as np
import h5py
from conf import conf
import traceback
import sys
from sgfmill import sgf
from play import (
    coord2index, make_play, game_init,
    show_board
)


class KGSSelfPlayWorker(Process):
    def __init__(self, process_id):
        Process.__init__(self, name='SelfPlayProcessor')
        self._process_id = process_id

    def run(self):
        try:
            SELF_PLAY_DIR = "../" + conf['SELF_PLAY_DIR']
            KGS_DATA_DIR = "../" + conf['KGS_DATA_DIR']
            game_dir = os.listdir(KGS_DATA_DIR)
            current_game = ""
            for file_name in game_dir:
                current_game = file_name
                game_name = file_name.split(".")[0]
                directory = os.path.join(SELF_PLAY_DIR, "KGS", game_name)
                if os.path.isdir(directory):
                    continue
                try:
                    os.makedirs(directory)
                except Exception:
                    print("Can not create dir ", directory)
                    continue

                game_data = self.play_game_kgs(os.path.join(KGS_DATA_DIR, file_name))
                moves = len(game_data['moves'])
                if moves == 0:  # empty data
                    print("No move generated! Remove dir ", directory)
                    os.rmdir(directory)
                    continue
                self.save_self_play_data(game_name, game_data)
                print("Finish self-play game", game_name)
        except Exception as e:
            print("EXCEPTION in KGSSelfPlayWorker!!!:", e, current_game)
            traceback.print_exc(file=sys.stdout)

    def setupHandicap(self, board, handicap):
        for move in handicap:
            board, _ = make_play(move[0], move[1], board, color=1) # always black do handicap
        return board

    def play_game_kgs(self, game_file):
        with open(game_file, "rb") as f:
            game = sgf.Sgf_game.from_bytes(f.read())
        winner = game.get_winner()
        board_size = game.get_size()
        root_node = game.get_root()
        b_player = root_node.get("PB")
        w_player = root_node.get("PW")

        print(game_file)
        print("board size ", board_size)
        print("Player B - W:", b_player, w_player)
        print("winner ", winner)

        SIZE = conf['SIZE']
        if board_size != SIZE:
            print("GAME SIZE IS NOT EXPECTED ", board_size)
            return

        board, _ = game_init()
        moves = []
        try:
            handicap = root_node.get("AB")  # Get handicap
            if len(handicap) > 0:
                board = self.setupHandicap(board, handicap)
        except:
            pass

        for move_n, node in enumerate(game.get_main_sequence()):
            print("move ", move_n)
            player, move = node.get_move()
            if player is None:
                continue
            if move is None:
                index = -1
                move = (board_size, board_size) #pass move
            else:
                index = coord2index(move[0], move[1])
            policy_target = np.zeros(board_size * board_size + 1, dtype=float)
            policy_target[index] = 1.0
            value = -1.0
            if winner == player:
                value = 1.0

            move_data = {
                'board': np.copy(board),
                'policy': policy_target,
                'value': value,
                'move': (move[0], move[1]),
                'move_n': move_n,
                'player': player
            }
            moves.append(move_data)
            board, _ = make_play(move[0], move[1], board, color=(1 if player == "b" else -1))

            # if conf['SHOW_EACH_MOVE']:
            #     print(show_board(board))

        game_data = {
            'moves': moves,
            'modelB_name': b_player,
            'modelW_name': w_player,
            'winner': winner,
        }
        return game_data

    def save_self_play_data(self, game_name, game_data):
        print("Saving ", game_name)
        SELF_PLAY_DIR = "../" + conf['SELF_PLAY_DIR']
        for move_data in game_data['moves']:
            board = move_data['board']
            policy_target = move_data['policy']
            value_target = move_data['value']
            move = move_data['move_n']
            directory = os.path.join(SELF_PLAY_DIR, "KGS", game_name, "move_%03d" % move)
            try:
                os.makedirs(directory)
            except:
                print("fail to make dir for kgs data")
            with h5py.File(os.path.join(directory, 'sample.h5'), 'w') as f:
                f.create_dataset('board', data=board, dtype=np.float32)
                f.create_dataset('policy_target', data=policy_target, dtype=np.float32)
                f.create_dataset('value_target', data=np.array(value_target), dtype=np.float32)

