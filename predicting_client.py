from multiprocessing.managers import BaseManager
import time

class PredictingClient(object):
    def __init__(self, remote_host='127.0.0.1', port=9999, password=b"secret"):
        # print("CLIENT INIT")
        BaseManager.register("predict_with_best_model")
        BaseManager.register("predict_with_latest_model")
        BaseManager.register("reload_model")
        BaseManager.register("get_best_model_name")
        BaseManager.register("get_latest_model_name")
        self.mgr = BaseManager(address=(remote_host, port), authkey=password)
        # try:
        self.mgr.connect()
        # except Exception as e:
        # print("CLIENT CONNECTED")

    def request_best_model_predict(self, board, is_symmetry=False):
        return self.mgr.predict_with_best_model(board, is_symmetry)._getvalue()

    def request_latest_model_predict(self, board, is_symmetry=False):
        return self.mgr.predict_with_latest_model(board, is_symmetry)._getvalue()

    def get_latest_model_name(self):
        return self.mgr.get_latest_model_name()._getvalue()

    def get_best_model_name(self):
        return self.mgr.get_best_model_name()._getvalue()

    def request_predict(self, model_indicator, board, is_symmetry=False):
        if model_indicator == "BEST_MODEL":
            return self.mgr.predict_with_best_model(board, is_symmetry)._getvalue()
        else:
            return self.mgr.predict_with_latest_model(board, is_symmetry)._getvalue()

if __name__ == "__main__":
    pc = PredictingClient()
    import numpy as np
    board = np.ones((1, 19, 19, 17), dtype=np.float32)
    # print(pc.get_latest_model_name())
    for i in range(10):
        p,v = pc.request_best_model_predict(board)
    # print(p)
    print(v[0][0])
