from multiprocessing import Process
from multiprocessing.managers import BaseManager
import os
from model import load_best_model, load_latest_model
from keras import backend as K
from symmetry import random_symmetry_predict

from app_log import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)


class PredictingWorker(Process):
    def __init__(self, gpuid, forever=True, port=9999,  password=b"secret"):
        Process.__init__(self, name='PredictingProcessor')
        self._gpuid = gpuid
        self._forever = forever
        self.port = port
        self.password = password
        self.latest_model = None
        self.best_model = None

    def __del__(self):
        print("CLEARING KERAS SESSION")
        K.clear_session()

    def load_model(self):
        self.best_model = load_best_model()
        self.latest_model = load_latest_model()

    def run(self):
        try:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)
            self.load_model()

            BaseManager.register("predict_with_best_model", self.predict_with_best_model)
            BaseManager.register("predict_with_latest_model", self.predict_with_latest_model)
            BaseManager.register("reload_model", self.load_model)
            BaseManager.register("get_best_model_name", self.get_best_model_name)
            BaseManager.register("get_latest_model_name", self.get_latest_model_name)
            mgr = BaseManager(address=('127.0.0.1', self.port), authkey=self.password)
            s = mgr.get_server()

            import numpy as np
            board = np.zeros((1,19,19,17), dtype=np.float32)
            self.predict_with_best_model(board)
            self.predict_with_latest_model(board)
            # print(p)
            # print(v)

            print("Listening for incoming connections...")
            s.serve_forever()
        except Exception as e:
            print(e)

    def predict_with_best_model(self, board, is_symmetry=False):
        if is_symmetry:
            return random_symmetry_predict(self.best_model, board)
        else:
            return self.best_model.predict_on_batch(board)

    def predict_with_latest_model(self, board, is_symmetry=False):
        if is_symmetry:
            return random_symmetry_predict(self.latest_model, board)
        else:
            return self.latest_model.predict_on_batch(board)

    def get_best_model_name(self):
        return self.best_model.name

    def get_latest_model_name(self):
        return self.latest_model.name