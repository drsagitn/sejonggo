from play import top_one_with_virtual_loss


def find_best_leaf_virtual_loss(node):
    moves = []
    while node['subtree'] != {}:
        action = top_one_with_virtual_loss(node)
        if action == {}:
            node['virtual_loss'] += 2
            node = node['parent']
            moves = moves[:-1]
            continue
        node = action['node']
        moves.append(action['action'])
    node['virtual_loss'] += 2
    return node, moves


def get_node_by_moves(node, moves):
    for m in moves:
        if node['subtree'].get(m) == None:
            raise Exception("ERROR: Unable to get node: Invalid moves array")
        node = node['subtree'][m]
    return node
