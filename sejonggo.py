from __future__ import print_function

from mcts1.tree_node import TreeNode
from mcts1.tree_search import *
from __init__ import __version__

spat_patterndict_file = conf['MSTC_PATTERN_FILE']
large_patterns_file = conf['LARGE_MCTS_PATTERN_FILE']


def gtp_io():
    """ GTP interface for our program.  We can play only on the board size
    which is configured (N), and we ignore color information and assume
    alternating play! """
    known_commands = ['boardsize', 'clear_board', 'komi', 'play', 'genmove',
                      'final_score', 'quit', 'name', 'version', 'known_command',
                      'list_commands', 'protocol_version', 'tsdebug']

    tree = TreeNode(pos=empty_position())
    tree.expand()
    print('Command list:', known_commands)
    print('Ready for command input!')
    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        if line == '':
            continue
        command = [s.lower() for s in line.split()]
        if re.match('\d+', command[0]):
            cmdid = command[0]
            command = command[1:]
        else:
            cmdid = ''
        owner_map = W*W*[0]
        ret = ''
        if command[0] == "boardsize":
            if int(command[1]) != N:
                print("Warning: Trying to set incompatible boardsize %s (!= %d)" % (command[1], N), file=sys.stderr)
                ret = None
        elif command[0] == "clear_board":
            tree = TreeNode(pos=empty_position())
            tree.expand()
        elif command[0] == "komi":
            # XXX: can we do this nicer?!
            tree.pos = Position(board=tree.pos.board, cap=(tree.pos.cap[0], tree.pos.cap[1]),
                                n=tree.pos.n, ko=tree.pos.ko, last=tree.pos.last, last2=tree.pos.last2,
                                komi=float(command[1]))
        elif command[0] == "play":
            c = parse_coord(command[2])
            if c is not None:
                # Find the next node in the game tree and proceed there
                if tree.children is not None and filter(lambda n: n.pos.last == c, tree.children):
                    tree = filter(lambda n: n.pos.last == c, tree.children).__next__()
                else:
                    # Several play commands in row, eye-filling move, etc.
                    tree = TreeNode(pos=tree.pos.move(c))

            else:
                # Pass move
                if tree.children[0].pos.last is None:
                    tree = tree.children[0]
                else:
                    tree = TreeNode(pos=tree.pos.pass_move())
        elif command[0] == "genmove":
            index = mcts_decision(policy, board, mcts_simulations, mcts_tree, temperature, model)
            str_coord(index)


            tree = tree_search(tree, conf['N_SIMS'], owner_map)
            if tree.pos.last is None:
                ret = 'pass'
            elif float(tree.w)/tree.v < conf['RESIGN_THRES']:
                ret = 'resign'
            else:
                ret = str_coord(tree.pos.last)
        elif command[0] == "final_score":
            score = tree.pos.score()
            if tree.pos.n % 2:
                score = -score
            if score == 0:
                ret = '0'
            elif score > 0:
                ret = 'B+%.1f' % (score,)
            elif score < 0:
                ret = 'W+%.1f' % (-score,)
        elif command[0] == "name":
            ret = 'Sejong Go Program'
        elif command[0] == "version":
            ret = 'v0.1'
        elif command[0] == "list_commands":
            ret = '\n'.join(known_commands)
        elif command[0] == "known_command":
            ret = 'true' if command[1] in known_commands else 'false'
        elif command[0] == "protocol_version":
            ret = '2'
        elif command[0] == "quit":
            print('=%s \n\n' % (cmdid,), end='')
            break
        else:
            print('Warning: Ignoring unknown command - %s' % (line,), file=sys.stderr)
            ret = None

        print_pos(tree.pos, sys.stderr, owner_map)
        if ret is not None:
            print('=%s %s\n\n' % (cmdid, ret,), end='')
        else:
            print('?%s ???\n\n' % (cmdid,), end='')
        sys.stdout.flush()


if __name__ == "__main__":
    print("Sejong-Go (v{})".format(__version__))
    if len(sys.argv) < 2:
        # Default action
        print('Starting in GTP mode...')
        gtp_io()
    elif len(sys.argv) > 2 and sys.argv[2] == "pattern":
        try:
            with open(spat_patterndict_file) as f:
                print('Loading pattern spatial dictionary...', file=sys.stderr)
                load_spat_patterndict(f)
            with open(large_patterns_file) as f:
                print('Loading large patterns...', file=sys.stderr)
                load_large_patterns(f)
            print('Done.', file=sys.stderr)
        except IOError as e:
            print('Warning: Cannot load pattern files: %s; will be much weaker, consider lowering EXPAND_VISITS 5->2' % (e,), file=sys.stderr)
    elif sys.argv[1] == "gtp":
        print('Starting in GTP mode...')
        gtp_io()
    # elif sys.argv[1] == "mcbenchmark":
    #     print(mcbenchmark(20))
    # elif sys.argv[1] == "tsbenchmark":
    #     t_start = time.time()
    #     print_pos(tree_search(TreeNode(pos=empty_position()), N_SIMS, W*W*[0], disp=False).pos)
    #     print('Tree search with %d playouts took %.3fs with %d threads; speed is %.3f playouts/thread/s' %
    #           (N_SIMS, time.time() - t_start, multiprocessing.cpu_count(),
    #            N_SIMS / ((time.time() - t_start) * multiprocessing.cpu_count())))
    # elif sys.argv[1] == "tsdebug":
    #     print_pos(tree_search(TreeNode(pos=empty_position()), N_SIMS, W*W*[0], disp=True).pos)
    else:
        print('Unknown action', file=sys.stderr)