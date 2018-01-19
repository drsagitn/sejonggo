import queue
from play import index2coord, make_play
from self_play import top_n_actions, new_subtree
from conf import conf
import threading

board_queue = queue.Queue()
subtree_queue = queue.Queue()


def init_workers():
    for i in range(conf['MCTS_BATCH_SIZE']):
        t1 = threading.Thread(target=board_worker)
        t1.daemon = True
        t1.start()

        t2 = threading.Thread(target=subtree_worker)
        t2.daemon = True
        t2.start()


def destroy_workers():
    for i in range(conf['MCTS_BATCH_SIZE']):
        board_queue.put((None, None, None, None, None, True))
        subtree_queue.put((None, None, None, None, None, None, True))


def board_worker():
    while True:
        dic, board, mcts_batch_size, boards, i, isStop = board_queue.get()
        if isStop:
            break

        action = dic['action']
        if dic['node']['subtree'] != {}:
            # already expanded
            tmp_node = dic['node']
            tmp_action = action
            x, y = index2coord(tmp_action)
            board, _ = make_play(x, y, board)
            n = 0
            while tmp_node['subtree'] != {}:
                tmp_max_actions = top_n_actions(tmp_node['subtree'], mcts_batch_size)
                tmp_d = tmp_max_actions[0]
                tmp_node = tmp_d['node']
                tmp_action = tmp_d['action']
                # The node for this action is the leaf, this is where the
                # update will start, working up the tree
                dic['node'] = tmp_node
                x, y = index2coord(tmp_action)
                n += 1
                make_play(x, y, board)
            boards[i] = board

        else:
            x, y = index2coord(action)
            make_play(x, y, board)
            boards[i] = board

        board_queue.task_done()


def subtree_worker():
    while True:
        node, policy, v, board, action, original_player, isStop = subtree_queue.get()
        if isStop:
            break

        shape = board.shape
        board = board.reshape([1] + list(shape))

        player = board[0, 0, 0, -1]
        # Inverse value if we're looking from other player perspective
        value = v[0] if player == original_player else -v[0]

        subtree = new_subtree(policy, board, node)
        leaf_node = action['node']
        leaf_node['subtree'] = subtree

        current_node = leaf_node
        while True:
            current_node['count'] += 1
            current_node['value'] += value
            current_node['mean_value'] = current_node['value'] / float(current_node['count'])
            if current_node['parent']:
                current_node = current_node['parent']
            else:
                break

        subtree_queue.task_done()