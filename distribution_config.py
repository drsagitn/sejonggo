from enum import Enum

import dbm

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
    db = dbm.open("events", 'c')
    if state == ASYNC_PIPELINE_STATE.SELF_PLAYING:
        db['self_play_event'] = '1'
        db['evaluate_event'] = '0'
    elif state == ASYNC_PIPELINE_STATE.EVALUATING:
        db['evaluate_event'] = '1'
        db['self_play_event'] = '0'
    db.close()


def set_slave_working(is_working):
    db = dbm.open("events", 'c')
    if is_working:
        db['slave_working'] = '1'
    else:
        db['slave_working'] = '0'
    db.close()


def is_slave_working():
    try:
        db = dbm.open("events", 'r')
        r = db['slave_working']
        db.close()
    except Exception as e:
        print("DB may not exist")
        print(e)
        return False
    if r == b'1':
        return True
    return False
