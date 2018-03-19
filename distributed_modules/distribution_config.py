from enum import Enum

import multiprocessing

self_play_event = multiprocessing.Event()
evaluate_event = multiprocessing.Event()
all_slaves_finished_event = multiprocessing.Event()
all_slaves_finished_event.set()


dconf = {
    "MASTER_IP": "211.180.114.12",
    "SLAVE_IPS": ["211.180.114.9"],
    'PORT': 8989,
    'REMOTE_PASSWORD': b'SejongGoPassword'
}
class ASYNC_PIPELINE_STATE(Enum):
    SELF_PLAYING = 1
    EVALUATING = 2
    TRAINING = 3
    OTHERS = 4


def turn_on_event(state):
    if state == ASYNC_PIPELINE_STATE.SELF_PLAYING:
        self_play_event.set()
        evaluate_event.clear()
    elif state == ASYNC_PIPELINE_STATE.EVALUATING:
        evaluate_event.set()
        self_play_event.clear()


# def turn_off_event(event):
#     if event == "SELF_PLAY":
#         self_play_event.clear()
#     elif event == "EVALUATE":
#         evaluate_event.clear()