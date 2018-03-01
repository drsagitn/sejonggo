from play import index2coord, make_play
from self_play import top_one_action, new_subtree
from conf import conf
from multiprocessing import Queue, Pool, Lock
from predicting_queue_worker import put_predict_request
board_queue = Queue()
subtree_queue = Queue()
simulation_result_queue = Queue()

N_SIMULATE_PROCESS = conf['N_SIMULATE_PROCESS']
process_pool = None
lock = None


def init_simulation_workers():
    global process_pool
    global lock
    lock = Lock()
    process_pool = Pool(processes=N_SIMULATE_PROCESS, initializer=init_pool_param, initargs=(lock,))


def init_pool_param(l):
    global lock
    lock = l


def destroy_simulation_workers():
    if process_pool is not None:
        process_pool.close()
        process_pool.join()


def basic_tasks2(node, board, moves, model_indicator, original_player):
    #  making board
    for m in moves:
        x,y = index2coord(m)
        board, _ = make_play(x,y, board)
    # predicting
    policy, value = put_predict_request(model_indicator, board)
    # subtree making
    node['subtree'] = new_subtree(policy, board, node)
    v = value if board[0, 0, 0, -1] == original_player else -value
    node['count'] += 1
    node['value'] += v
    node['mean_value'] = node['value'] / float(node['count'])
    simulation_result_queue.put((node, moves))


def basic_tasks(node, board, move, model_indicator, original_player):
    moves = [move]
    while node['subtree'] != {}:
        action = top_one_action(node['subtree'])
        node = action['node']
        moves.append(action['action'])
    #  making board
    for m in moves:
        x,y = index2coord(m)
        board, _ = make_play(x,y, board)
    policy, value = put_predict_request(model_indicator, board)
    node['subtree'] = new_subtree(policy, board, node)

    # backpropagation
    v = value if board[0, 0, 0, -1] == original_player else -value
    while True:
        node['count'] += 1
        node['value'] += v
        node['mean_value'] = node['value'] / float(node['count'])
        if node['parent']:
            node = node['parent']
        else:
            break
    return node

def board_worker(input_tuple):
    try:
        dic, board = input_tuple
        action = dic['action']
        if dic['node']['subtree'] != {}:
            tmp_node = dic['node']
            tmp_action = action
            x, y = index2coord(tmp_action)
            board, _ = make_play(x, y, board)
            while tmp_node['subtree'] != {}:
                tmp_d = top_one_action(tmp_node['subtree'])
                tmp_node = tmp_d['node']
                tmp_action = tmp_d['action']
                dic['node'] = tmp_node
                x, y = index2coord(tmp_action)
                make_play(x, y, board)
            return board[0]

        else:
            x, y = index2coord(action)
            make_play(x, y, board)
            return board[0]
    except Exception:
        print("EXCEPTION IN BOARD WORKER!!!!!!")


def subtree_worker(input_tuple):
    try:
        policy, board = input_tuple
        shape = board.shape
        board = board.reshape([1] + list(shape))
        subtree = new_subtree(policy, board, None)
        return subtree
    except Exception:
        print("EXCEPTION IN SUBTREE WORKER!!!!!!")
