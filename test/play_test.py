import sys
sys.path.append("..")
from model import *
from self_play import play_game, self_play, save_file
from conf import conf
from tqdm import tqdm


MCTS_SIMULATIONS = conf['MCTS_SIMULATIONS']


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Missing parameters. 3 parameters required to start game play: num_of_game, model1_name, model2_name')

    else:
        model1_win = 0
        model2_win = 0
        num_games = sys.argv[1]
        model1_name = sys.argv[2]
        model2_name = sys.argv[3]
        print("Loading models...")
        model1 = load_model(os.path.join('..', conf['MODEL_DIR'], model1_name), custom_objects={'loss': loss})
        print("Loaded model", model1_name)
        model2 = load_model(os.path.join('..', conf['MODEL_DIR'], model2_name), custom_objects={'loss': loss})
        print("Loaded model", model2_name)
        desc = "Duel %s vs %s" % (model1.name, model2.name)
        tq = tqdm(range(int(num_games)), desc=desc)
        for game in tq:
            boards_and_policies, winner, winner_model = play_game(model1, model2, MCTS_SIMULATIONS,
                                                                  stop_exploration=0)
            if winner_model == model1:
                model1_win += 1
            else:
                model2_win += 1
            new_desc = desc + " (%s %d - %d %s)" % (model1_name, model1_win, model2_win, model2_name)
            tq.set_description(new_desc)




