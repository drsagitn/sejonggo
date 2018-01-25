from play import index2coord, make_play
from self_play import top_one_action, new_subtree
from conf import conf
from multiprocessing import Process, Queue, Pool


board_queue = Queue()
subtree_queue = Queue()

MCTS_BATCH_SIZE = conf['MCTS_BATCH_SIZE']
process_pool = None


def init_workers():
    global process_pool
    process_pool = Pool(processes=MCTS_BATCH_SIZE)


def destroy_workers():
    if process_pool is not None:
        process_pool.close()
        process_pool.join()


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
