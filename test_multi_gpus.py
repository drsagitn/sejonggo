import tensorflow as tf
import concurrent.futures
import os
from model import create_initial_model, load_latest_model, load_best_model
from keras import backend as K
from train import train
from conf import conf
from evaluator import evaluate
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def init_directories():
    try:
        os.mkdir(conf['MODEL_DIR'])
    except:
        pass
    try:
        os.mkdir(conf['LOG_DIR'])
    except:
        pass


def train_with_gpu(model, game_model_name, gpu):
    with tf.device(gpu):
        train(model, game_model_name=game_model_name)


def evaluate_with_gpu(best_model, model, gpu):
    with tf.device(gpu):
        evaluate(best_model, model)

def main():
    init_directories()
    model = load_latest_model()
    best_model = load_best_model()

    gpus = get_available_gpus()


    while True:
        with concurrent.futures.ProcessPoolExecutor(len(gpus)) as executor:
            executor.submit(train_with_gpu, model, best_model.name, gpus[0])
            executor.submit(evaluate, best_model, model, gpus[1])
        K.clear_session()


if __name__ == "__main__":
    main()
