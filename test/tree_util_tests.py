import unittest
from play import tree_depth
from tree_util import find_best_leaf_virtual_loss, get_node_by_moves


class TreeTestCase(unittest.TestCase):
    def setUp(self):
        tree = {
            'id': 1,
            'count': 0,
            'mean_value': 0,
            'virtual_loss': 0,
            'value': 0,
            'parent': None,
            'subtree': {
                0: {
                    'id': 2,
                    'count': 0,
                    'p': 1,
                    'value': 1,
                    'mean_value': 0,
                    'virtual_loss': 0,
                    'subtree': {
                        3: {
                            'id': 4,
                            'count': 0,
                            'p': 1,
                            'value': 1,
                            'mean_value': 0,
                            'virtual_loss': 0,
                            'subtree': {}
                        },
                        4: {
                            'id': 5,
                            'count': 0,
                            'p': 0,
                            'mean_value': 0,
                            'virtual_loss': 0,
                            'value': 0,
                            'subtree': {}
                        }
                    }
                },
                1: {
                    'id': 3,
                    'count': 0,
                    'p': 0,
                    'mean_value': 0,
                    'virtual_loss': 0,
                    'value': 0,
                    'subtree': {}
                }
            }
        }
        tree['subtree'][0]['parent'] = tree
        tree['subtree'][1]['parent'] = tree
        tree['subtree'][0]['subtree'][3]['parent'] = tree['subtree'][0]
        tree['subtree'][0]['subtree'][4]['parent'] = tree['subtree'][0]
        self.tree = tree

    def tearDown(self):
        return

    def test_tree_depth(self):
        d = tree_depth(self.tree)
        self.assertEqual(d, 3)

    def test_find_best_leaf(self):
        node, move = find_best_leaf_virtual_loss(self.tree)
        self.assertEqual(node['id'], 4)
        self.assertGreater(node['virtual_loss'], 0)
        self.assertEqual(move, [0, 3])

        node2, move2 = find_best_leaf_virtual_loss(self.tree)
        self.assertEqual(node2['id'], 5)
        self.assertGreater(node2['virtual_loss'], 0)
        self.assertEqual(move2, [0, 4])

        node3, move3 = find_best_leaf_virtual_loss(self.tree)
        self.assertEqual(node3['id'], 3)
        self.assertGreater(node3['virtual_loss'], 0)
        self.assertGreater(self.tree['subtree'][0]['virtual_loss'], 0)
        self.assertEqual(move3, [1])

    def test_all_leaf_busy(self):
        tree = {
            'id': 1,
            'count': 0,
            'mean_value': 0,
            'virtual_loss': 0,
            'value': 0,
            'parent': None,
            'subtree': {
                0: {
                    'id': 2,
                    'count': 0,
                    'p': 1,
                    'value': 1,
                    'mean_value': 0,
                    'virtual_loss': 0,
                    'subtree': {
                    }
                },
                1: {
                    'id': 3,
                    'count': 0,
                    'p': 0,
                    'mean_value': 0,
                    'virtual_loss': 0,
                    'value': 0,
                    'subtree': {}
                }
            }
        }
        tree['subtree'][0]['parent'] = tree
        tree['subtree'][1]['parent'] = tree
        _, _ = find_best_leaf_virtual_loss(tree)
        _, _ = find_best_leaf_virtual_loss(tree)
        node3, move3 = find_best_leaf_virtual_loss(tree)
        self.assertEqual(node3, None)
        self.assertEqual(move3, None)



    def test_get_leaf_by_moves(self):
        des = get_node_by_moves(self.tree, [0])
        self.assertEqual(des['id'], 2)

        des = get_node_by_moves(self.tree, [0, 4])
        self.assertEqual(des['id'], 5)

        des = get_node_by_moves(self.tree, [1])
        self.assertEqual(des['id'], 3)

        with self.assertRaises(Exception):
            get_node_by_moves(self.tree, [0, 8])
