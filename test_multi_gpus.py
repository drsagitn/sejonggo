# import tensorflow as tf
# import concurrent.futures
# import os
# from model import create_initial_model, load_latest_model, load_best_model
# from keras import backend as K
# from train import train
# from conf import conf
# from evaluator import evaluate
# from tensorflow.python.client import device_lib
# import math
from train_worker import *
from selfplay_worker import *
from evaluate_worker import *
import logging
from app_log import setup_logging
setup_logging()
logger = logging.getLogger(__name__)
#
#
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
#
# def init_directories():
#     try:
#         os.mkdir(conf['MODEL_DIR'])
#     except:
#         pass
#     try:
#         os.mkdir(conf['LOG_DIR'])
#     except:
#         pass
#
#
# def train_with_gpu(gpu):
#     logger.info('Start train with %s', gpu)
#     config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
#     sess = tf.Session(config=config)
#     K.set_session(sess)
#     with sess.graph.as_default():
#         with tf.device(gpu):
#             logger.info('Start load latest model...')
#             model = load_latest_model()
#             logger.info('Latest model %s loaded', model.name)
#             logger.info('Start training...')
#             train(model, game_model_name="best_model.h5")
#             logger.info('Finished training')
#         K.clear_session()
#
# def is_prime(n, gpu):
#     logger.info('Start train with %s', gpu)
#     with tf.device(gpu):
#         if n % 2 == 0:
#             return False
#
#         sqrt_n = int(math.floor(math.sqrt(n)))
#         for i in range(3, sqrt_n + 1, 2):
#             if n % i == 0:
#                 return False
#         return True
#
# def evaluate_with_gpu(best_model, model, gpu):
#     with tf.device(gpu):
#         evaluate(best_model, model)
#
#
# def run_session(device):
#     gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=device)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#     print('Using device #%s' % device)
#     a = tf.placeholder(tf.int16, name='a')
#     y = tf.identity(a, name='y')
#     print(sess.run(y, feed_dict={a: 3}))
#     sess.close()
#     print('Done.')


def main():
    #run_session('0')
    #run_session('1')

    #init_directories()
    #gpus = get_available_gpus()

    #with concurrent.futures.ThreadPoolExecutor(len(gpus)) as executor:
    #    executor.submit(train_with_gpu, gpus[0])

    # print(statistic("games", 200))
    #clean_up("self_play_data", 11)
    #print(statistic("self_play_data", 200))
    # logger.info("Clean up self-play folder")
    # clean_up("self_play_data")
    #
    workers = list()
    # #workers.append(TrainWorker(0))
    workers.append(SelfPlayWorker(0, forever=False))
    workers.append(SelfPlayWorker(1, forever=False))
    workers.append(SelfPlayWorker(2, forever=False))
    workers.append(SelfPlayWorker(3, forever=False))
    # workers.append(EvaluateWorker(0, forever=False))
    # workers.append(SelfPlayWorker(1))

    start = datetime.datetime.now()
    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()
    stop = datetime.datetime.now()
    print("Total time:", (stop - start).seconds)





if __name__ == "__main__":
    main()
