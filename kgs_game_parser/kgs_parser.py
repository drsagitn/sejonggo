from conf import conf
from kgs_game_parser.KGSSelfPlayWorker import KGSSelfPlayWorker
from sgfmill import sgf
import os
import h5py

def test():
    with open("kgs_data_small/Woods-Sunako.sgf", "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    winner = game.get_winner()
    print("winner: ", winner)
    board_size = game.get_size()
    print("board size: ", board_size)
    root_node = game.get_root()
    b_player = root_node.get("PB")
    w_player = root_node.get("PW")
    for idx, node in enumerate(game.get_main_sequence()):
        print(idx, ":",node.get_move())

def unit_test(game_name):
    with open("kgs_data_small/" + game_name + ".sgf", "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    SELF_PLAY_DATA_DIR = "sp_self_play_data/KGS/"
    for j, move_folder in enumerate(sorted(os.listdir(SELF_PLAY_DATA_DIR + game_name))):
        with h5py.File(os.path.join(SELF_PLAY_DATA_DIR, game_name, move_folder, "sample.h5")) as f:
            board = f['board'][:]
            policy = f['policy_target'][:]
            value_target = f['value_target'][()]


def clean_up_empty():
    try:
        dirs_to_clean = [conf['SELF_PLAY_DIR']]
        for folder in dirs_to_clean:
            for _dir in os.listdir(folder):
                dir_path = os.path.join(folder, _dir)
                for d in os.listdir(dir_path):
                    d_path = os.path.join(dir_path, d)
                    if len(os.listdir(d_path)) == 0:
                        print("Clean up empty dir", d_path)
                        os.rmdir(d_path)
    except Exception as e:
        print("EXCEPTION WHILE CLEANING FOLDERS!!")
        print(e)

def main():
    clean_up_empty()
    workers = [KGSSelfPlayWorker(i) for i in range(5)]
    for p in workers: p.start()
    for p in workers: p.join()


if __name__ == "__main__":
    # test()
    # unit_test("waage-ben0-2")
    main()