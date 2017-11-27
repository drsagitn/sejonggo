from math import sqrt

conf = {
    'MODEL_DIR': 'models',
    'LOG_DIR': 'logs',
    'BEST_MODEL': 'best_model.h5',
    'SHOW_EACH_MOVE': False,
    'SHOW_END_GAME': False,
    'REPORT_PERIOD': 200,
    'GAMES_DIR': 'games',

    ### MODEL ###
    'N_RESIDUAL_BLOCKS': 20,  # Size of the tower of residual blocks, 20 for small model, 40 for alphagozero full size
    'L2_EPSILON': 1e-4,  # The epsilon coefficient in the loss value function, 1e-4 in paper

    ### SELF-PLAY ###
    'N_GAMES': 250,  # Number of games of self play generated by best_model, 25k in paper
    'MCTS_SIMULATIONS': 512,  # 1.6k in paper
    'SIZE': 9,  # board size 19 in paper
    'KOMI': 5.5,  # The komi points given to white player
    'STOP_EXPLORATION': 30,  # Number of plays after which temperature goes to 0 , 30 in paper
    'MCTS_BATCH_SIZE': 8,  # Size of the prediction batch while exploring mcts
    'DIRICHLET_ALPHA': .03,  # The value of dirichlet coefficient in the nois of root_node of mcts simulation
    'DIRICHLET_EPSILON': .25,  # How much the noise is accounted for
    'RESIGNATION_PERCENT': .10, # 10% of the time we don't use resignation to assess resignation value
    'RESIGNATION_ALLOWED_ERROR': .05,  # 5% of the time we resign a game we could have won

    ### TRAIN ###
    'TRAIN_BATCH_SIZE': 32,  # Batch size in the training phase, 32 in paper
    'EPOCHS_PER_SAVE': 100,  # A model will be saved to be evaluated this amount of epochs, 1000 in paper
    'NUM_WORKERS': 64,  # We use this many GPU workers so split the task, 64 in paper
    'HISTOGRAM_FREQ': 0,  # Shows the histograms in Tensorboard. For debugging
    'VALIDATION_SPLIT': 0,  # Needed if you want histograms in Tensorboard.

    ### EVALUATOR ###
    'EVALUATE_N_GAMES': 100,  # The number of games to test on to elect new best model, 400 in paper
    'EVALUATE_MARGIN': .5 + 1 / sqrt(100),  # Model has to win by that margin to be elected, 55% in paper
    'SGF_ENABLED': True,

    ### MCTS
    'MSTC_PATTERN_FILE': 'patterns.spat',
    'LARGE_MCTS_PATTERN_FILE': 'patterns.prob',
    'PRIOR_EVEN': 10,
    'RESIGN_THRES': 0.2,
    'N_SIMS': 1400,
    'EXPAND_VISITS': 8,
    'FASTPLAY20_THRES': 0.8,  # if at 20% playouts winrate is >this, stop reading
    'FASTPLAY5_THRES': 0.95,  # if at 5% playouts winrate is >this, stop reading
    'PROB_HEURISTIC': {'capture': 0.9, 'pat3': 0.95},  # probability of heuristic suggestions being taken in playout
    'PROB_SSAREJECT': 0.9,  # probability of rejecting suggested self-atari in playout
    'PROB_RSAREJECT': 0.5,
# probability of rejecting random self-atari in playout; this is lower than above to allow nakade
    'PRIOR_EVEN': 10,  # should be even number; 0.5 prior
    'PRIOR_SELFATARI': 10,  # negative prior
    'PRIOR_CAPTURE_ONE': 15,
    'PRIOR_CAPTURE_MANY': 30,
    'PRIOR_PAT3': 10,
    'PRIOR_LARGEPATTERN': 100,  # most moves have relatively small probability
    'PRIOR_CFG': [24, 22, 8],  # priors for moves in cfg dist. 1, 2, 3
    'PRIOR_EMPTYAREA': 10,
    'RAVE_EQUIV': 3500,
}
