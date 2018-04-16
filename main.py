import os
from model import create_initial_model, load_latest_model, load_best_model
from keras import backend as K
from train import train
from conf import conf
from evaluator import evaluate
from __init__ import __version__
from app_log import setup_logging
from utils import init_directories
from simulation_workers import init_simulation_workers
from model import load_model_by_name

def main():
    print("Starting run (v{})".format(__version__))
    init_directories()
    if conf['THREAD_SIMULATION']:
        init_simulation_workers()
    model_name = "model_1"
    model = create_initial_model(name=model_name)


    while True:
        model = load_latest_model()
        best_model = load_best_model()
        train(model, game_model_name=best_model.name)
        evaluate(best_model, model)
        K.clear_session()


def compare_model(model_name_1, model_name_2):
    model1 = load_model_by_name(model_name_1)
    model2 = load_model_by_name(model_name_2)
    model1.summary()
    model1.save("temp_model.h5")
    print("=======================")
    model2.summary()
    print("=======================")
    w1 = model1.get_weights()
    print(w1[3])
    print("+++++++++++++++++++++++++")
    w2 = model2.get_weights()
    print(w2[3])




if __name__ == "__main__":
    compare_model("model_142.h5", "temp_model.h5")
    # main()
