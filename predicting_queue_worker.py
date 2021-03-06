from multiprocessing import Process, Queue, Pipe
from model import load_best_model, load_latest_model
from symmetry import random_symmetry_predict
import os
import time
import numpy as np
from conf import conf


board_queue = Queue()

def init_predicting_workers(GPUs):
    for GPU_id in GPUs:
        p = PredictingQueueWorker(GPU_id)
        p.start()


def destroy_predicting_workers(GPUs):
    for _ in GPUs:
        board_queue.put((None, None, None, None))


class PredictingQueueWorker(Process):
    def __init__(self, gpu_id):
        Process.__init__(self, name='PredictingQueueWorker')
        self.latest_model = None
        self.best_model = None
        self.gpu_id = gpu_id

    # def __del__(self):
        # print("CLEARING KERAS SESSION")
        # K.clear_session()

    def load_model(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        self.best_model = load_best_model()
        self.latest_model = load_latest_model()

    def run(self):
        try:
            self.load_model()
            while True:
                if board_queue.empty():
                    time.sleep(0.1)
                root = {"BEST_SYM":{'board':[], 'a':[]},
                        "LATEST_SYM":{'board':[], 'a':[]},
                        "BEST":{'board':[], 'a':[]},
                        "LATEST":{'board':[], 'a':[]},
                        "BEST_NAME" : {'a':[]},
                        "LATEST_NAME": {'a': []}
                        }
                n = conf['PREDICTING_BATCH_SIZE']
                while n > 0 and not board_queue.empty():
                    try:
                        board, indicator, a, response_now = board_queue.get_nowait()
                        if a is None and indicator is None and board is None:
                            print("SHUTING DONW PREDICTING WORKER!!!")
                            # K.clear_session()
                            return
                        root[indicator]['a'].append(a)
                        current_boards = root[indicator].get('board')
                        if current_boards is None:
                            break
                        if current_boards == []:
                            root[indicator]['board'] = board
                        else:
                            root[indicator]['board'] = np.vstack((current_boards, board))
                        if response_now:
                            break
                        n = n - 1
                    except Exception:
                        pass

                if root["BEST"]['board'] != []:
                    p, v = self.best_model.predict_on_batch(root["BEST"]['board'])
                    for index, a in enumerate(root["BEST"]['a']):
                        a.send((p[index], v[index][0]))
                if root["LATEST"]['board'] != []:
                    p, v = self.latest_model.predict_on_batch(root["LATEST"]['board'])
                    for index, a in enumerate(root["LATEST"]['a']):
                        a.send((p[index], v[index][0]))
                if root["BEST_SYM"]['board'] != []:
                    # tt = len(root["BEST_SYM"]['board'])
                    # print(tt)
                    # total += tt
                    # print("%s..%s" % (tt, total))
                    p, v = random_symmetry_predict(self.best_model, root["BEST_SYM"]['board'])
                    for index, a in enumerate(root["BEST_SYM"]['a']):
                        a.send((p[index], v[index][0]))
                if root["LATEST_SYM"]['board'] != []:
                    p, v = random_symmetry_predict(self.best_model, root["LATEST_SYM"]['board'])
                    for index, a in enumerate(root["LATEST_SYM"]['a']):
                        a.send((p[index], v[index][0]))
                if root["LATEST_NAME"]['a'] != []:
                    name = self.latest_model.name
                    for index, a in enumerate(root["LATEST_NAME"]['a']):
                        a.send(name)
                if root["BEST_NAME"]['a'] != []:
                    name = self.best_model.name
                    for index, a in enumerate(root["BEST_NAME"]['a']):
                        a.send(name)

        except Exception as e:
            print("PREDICTING QUEUE WORKER EXCEPTION !!!")
            print(e)


def put_name_request(model_indicator):
    if model_indicator == "BEST_SYM" or model_indicator == "BEST":
        model_indicator = "BEST_NAME"
    elif model_indicator == "LATEST_SYM" or model_indicator == "LATEST":
        model_indicator = "LATEST_NAME"
    a, b = Pipe()
    board_queue.put((None, model_indicator, a, True))
    name = b.recv()
    return name


def put_predict_request(model_indicator, board, response_now=False):
    a, b = Pipe()
    board_queue.put((board, model_indicator, a, response_now))
    p, v = b.recv()
    return p, v


if __name__ == "__main__":
    init_predicting_workers(0)
    time.sleep(15)

    board = np.ones((1, 19, 19, 17), dtype=np.float32)
    import datetime
    # start = datetime.datetime.now()
    # [board_queue.put(boards) for _ in range(10)]
    # result_queue.get()
    # end = datetime.datetime.now()
    # print("END TIME: %s" % end)
    # print("START TIME: %s" % start)
    # print("TOTAL TIME: %s" % (end-start))

    workers = [Process(target=put_predict_request, args=("LATEST",board)) for _ in range(1)]
    start = datetime.datetime.now()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    end = datetime.datetime.now()
    print("TOTAL TIME: %s" % (end - start))