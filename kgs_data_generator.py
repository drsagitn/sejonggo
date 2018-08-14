import numpy as np
import keras
from conf import conf
from sgfmill import sgf
from play import (
    coord2index, make_play, game_init
)
import os
import patoolib
import shutil
from sklearn.model_selection import train_test_split


def get_KGS_training_desc():
    all_files = []
    data_dir = os.path.join(conf['SELF_PLAY_DIR'], "KGS")
    print("READING KGS data from ", data_dir)
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            full_filename = os.path.join(root, f)
            all_files.append(full_filename)

    x_train, x_test = train_test_split(all_files, test_size=0.3, random_state=2)
    return {'train': x_train, 'validation': x_test}

class KGSDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        # self.ziplist = self.get_ziplist()
        self.gamelist = self.get_gamelist()
        self.movelist = []

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(400000000 / self.batch_size)

    def __getitem__(self, index):
        X, y = self.__data_generation(self.get_movelist_by_batchsize())
        return X, y

    def __data_generation(self, move_list_batch):
        SIZE = conf['SIZE']
        X = np.zeros((self.batch_size, *self.dim))
        policy_y = np.zeros((self.batch_size, 1))
        value_y = np.zeros((self.batch_size, SIZE * SIZE + 1))
        for j, item in enumerate(move_list_batch):
            board = item['board']
            policy = item['policy']
            value_target = item['value']
            X[j] = board
            policy_y[j] = value_target
            value_y[j] = policy

        return X, [value_y, policy_y]


    ##  ZIPFOLDER  >>  ZIP-LIST  >>  GAME-LIST  >>  MOVE-LIST

    def get_ziplist(self):
        zip_folder = conf['KGS_ZIP_FOLDER']
        return os.listdir(zip_folder)

    def get_gamelist(self):
        data_dir = conf['KGS_DATA_DIR']
        return_arr = []
        print("Collecting game files...")
        for f in os.listdir(data_dir):
            return_arr.append(os.path.join(data_dir, f))
        return return_arr

    def get_movelist(self):
        game_file = self.gamelist.pop()
        return self.play_game_kgs(os.path.join(game_file))

    def get_movelist_by_batchsize(self):
        while len(self.movelist) < self.batch_size*8:
            self.movelist = self.movelist + self.get_movelist()
            if self.movelist is None:
                return None
        return [self.movelist.pop() for _ in range(self.batch_size)]

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

        return moves