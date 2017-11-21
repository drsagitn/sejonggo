from conf import conf
import re
from functools import reduce
from collections import namedtuple
import random

spat_patterndict_file = conf['MSTC_PATTERN_FILE']
large_patterns_file = conf['LARGE_MCTS_PATTERN_FILE']
N = conf['SIZE']
W = N + 2
empty = "\n".join([(N+1)*' '] + N*[' '+N*'.'] + [(N+2)*' '])
pat_gridcular_seq = [  # Sequence of coordinate offsets of progressively wider diameters in gridcular metric
        [[0,0],
         [0,1], [0,-1], [1,0], [-1,0],
         [1,1], [-1,1], [1,-1], [-1,-1], ],  # d=1,2 is not considered separately
        [[0,2], [0,-2], [2,0], [-2,0], ],
        [[1,2], [-1,2], [1,-2], [-1,-2], [2,1], [-2,1], [2,-1], [-2,-1], ],
        [[0,3], [0,-3], [2,2], [-2,2], [2,-2], [-2,-2], [3,0], [-3,0], ],
        [[1,3], [-1,3], [1,-3], [-1,-3], [3,1], [-3,1], [3,-1], [-3,-1], ],
        [[0,4], [0,-4], [2,3], [-2,3], [2,-3], [-2,-3], [3,2], [-3,2], [3,-2], [-3,-2], [4,0], [-4,0], ],
        [[1,4], [-1,4], [1,-4], [-1,-4], [3,3], [-3,3], [3,-3], [-3,-3], [4,1], [-4,1], [4,-1], [-4,-1], ],
        [[0,5], [0,-5], [2,4], [-2,4], [2,-4], [-2,-4], [4,2], [-4,2], [4,-2], [-4,-2], [5,0], [-5,0], ],
        [[1,5], [-1,5], [1,-5], [-1,-5], [3,4], [-3,4], [3,-4], [-3,-4], [4,3], [-4,3], [4,-3], [-4,-3], [5,1], [-5,1], [5,-1], [-5,-1], ],
        [[0,6], [0,-6], [2,5], [-2,5], [2,-5], [-2,-5], [4,4], [-4,4], [4,-4], [-4,-4], [5,2], [-5,2], [5,-2], [-5,-2], [6,0], [-6,0], ],
        [[1,6], [-1,6], [1,-6], [-1,-6], [3,5], [-3,5], [3,-5], [-3,-5], [5,3], [-5,3], [5,-3], [-5,-3], [6,1], [-6,1], [6,-1], [-6,-1], ],
        [[0,7], [0,-7], [2,6], [-2,6], [2,-6], [-2,-6], [4,5], [-4,5], [4,-5], [-4,-5], [5,4], [-5,4], [5,-4], [-5,-4], [6,2], [-6,2], [6,-2], [-6,-2], [7,0], [-7,0], ],
    ]

pat3src = [  # 3x3 playout patterns; X,O are colors, x,o are their inverses
       ["XOX",  # hane pattern - enclosing hane
        "...",
        "???"],
       ["XO.",  # hane pattern - non-cutting hane
        "...",
        "?.?"],
       ["XO?",  # hane pattern - magari
        "X..",
        "x.?"],
       # ["XOO",  # hane pattern - thin hane
       #  "...",
       #  "?.?", "X",  - only for the X player
       [".O.",  # generic pattern - katatsuke or diagonal attachment; similar to magari
        "X..",
        "..."],
       ["XO?",  # cut1 pattern (kiri] - unprotected cut
        "O.o",
        "?o?"],
       ["XO?",  # cut1 pattern (kiri] - peeped cut
        "O.X",
        "???"],
       ["?X?",  # cut2 pattern (de]
        "O.O",
        "ooo"],
       ["OX?",  # cut keima
        "o.O",
        "???"],
       ["X.?",  # side pattern - chase
        "O.?",
        "   "],
       ["OX?",  # side pattern - block side cut
        "X.O",
        "   "],
       ["?X?",  # side pattern - block side connection
        "x.O",
        "   "],
       ["?XO",  # side pattern - sagari
        "x.x",
        "   "],
       ["?OX",  # side pattern - cut
        "X.O",
        "   "],
       ]




# 3x3 pattern routines (those patterns stored in pat3src above)

def pat3_expand(pat):
    """ All possible neighborhood configurations matching a given pattern;
    used just for a combinatoric explosion when loading them in an
    in-memory set. """
    def pat_rot90(p):
        return [p[2][0] + p[1][0] + p[0][0], p[2][1] + p[1][1] + p[0][1], p[2][2] + p[1][2] + p[0][2]]
    def pat_vertflip(p):
        return [p[2], p[1], p[0]]
    def pat_horizflip(p):
        return [l[::-1] for l in p]
    def pat_swapcolors(p):
        return [l.replace('X', 'Z').replace('x', 'z').replace('O', 'X').replace('o', 'x').replace('Z', 'O').replace('z', 'o') for l in p]
    def pat_wildexp(p, c, to):
        i = p.find(c)
        if i == -1:
            return [p]
        return reduce(lambda a, b: a + b, [pat_wildexp(p[:i] + t + p[i+1:], c, to) for t in to])
    def pat_wildcards(pat):
        return [p for p in pat_wildexp(pat, '?', list('.XO '))
                  for p in pat_wildexp(p, 'x', list('.O '))
                  for p in pat_wildexp(p, 'o', list('.X '))]
    return [p for p in [pat, pat_rot90(pat)]
              for p in [p, pat_vertflip(p)]
              for p in [p, pat_horizflip(p)]
              for p in [p, pat_swapcolors(p)]
              for p in pat_wildcards(''.join(p))]


pat3set = set([p.replace('O', 'x') for p in pat3src for p in pat3_expand(p)])


def neighborhood_33(board, c):
    """ return a string containing the 9 points forming 3x3 square around
    a certain move candidate """
    return (board[c-W-1 : c-W+2] + board[c-1 : c+2] + board[c+W-1 : c+W+2]).replace('\n', ' ')


def fix_atari(pos, c, singlept_ok=False, twolib_test=True, twolib_edgeonly=False):
    """ An atari/capture analysis routine that checks the group at c,
    determining whether (i) it is in atari (ii) if it can escape it,
    either by playing on its liberty or counter-capturing another group.

    N.B. this is maybe the most complicated part of the whole program (sadly);
    feel free to just TREAT IT AS A BLACK-BOX, it's not really that
    interesting!

    The return value is a tuple of (boolean, [coord..]), indicating whether
    the group is in atari and how to escape/capture (or [] if impossible).
    (Note that (False, [...]) is possible in case the group can be captured
    in a ladder - it is not in atari but some capture attack/defense moves
    are available.)

    singlept_ok means that we will not try to save one-point groups;
    twolib_test means that we will check for 2-liberty groups which are
    threatened by a ladder
    twolib_edgeonly means that we will check the 2-liberty groups only
    at the board edge, allowing check of the most common short ladders
    even in the playouts """

    def read_ladder_attack(pos, c, l1, l2):
        """ check if a capturable ladder is being pulled out at c and return
        a move that continues it in that case; expects its two liberties as
        l1, l2  (in fact, this is a general 2-lib capture exhaustive solver) """
        for l in [l1, l2]:
            pos_l = pos.move(l)
            if pos_l is None:
                continue
            # fix_atari() will recursively call read_ladder_attack() back;
            # however, ignore 2lib groups as we don't have time to chase them
            is_atari, atari_escape = fix_atari(pos_l, c, twolib_test=False)
            if is_atari and not atari_escape:
                return l
        return None

    fboard = floodfill(pos.board, c)
    group_size = fboard.count('#')
    if singlept_ok and group_size == 1:
        return (False, [])
    # Find a liberty
    l = contact(fboard, '.')
    # Ok, any other liberty?
    fboard = board_put(fboard, l, 'L')
    l2 = contact(fboard, '.')
    if l2 is not None:
        # At least two liberty group...
        if twolib_test and group_size > 1 \
           and (not twolib_edgeonly or line_height(l) == 0 and line_height(l2) == 0) \
           and contact(board_put(fboard, l2, 'L'), '.') is None:
            # Exactly two liberty group with more than one stone.  Check
            # that it cannot be caught in a working ladder; if it can,
            # that's as good as in atari, a capture threat.
            # (Almost - N/A for countercaptures.)
            ladder_attack = read_ladder_attack(pos, c, l, l2)
            if ladder_attack:
                return (False, [ladder_attack])
        return (False, [])

    # In atari! If it's the opponent's group, that's enough...
    if pos.board[c] == 'x':
        return (True, [l])

    solutions = []

    # Before thinking about defense, what about counter-capturing
    # a neighboring group?
    ccboard = fboard
    while True:
        othergroup = contact(ccboard, 'x')
        if othergroup is None:
            break
        a, ccls = fix_atari(pos, othergroup, twolib_test=False)
        if a and ccls:
            solutions += ccls
        # XXX: floodfill is better for big groups
        ccboard = board_put(ccboard, othergroup, '%')

    # We are escaping.  Will playing our last liberty gain
    # at least two liberties?  Re-floodfill to account for connecting
    escpos = pos.move(l)
    if escpos is None:
        return (True, solutions)  # oops, suicidal move
    fboard = floodfill(escpos.board, l)
    l_new = contact(fboard, '.')
    fboard = board_put(fboard, l_new, 'L')
    l_new_2 = contact(fboard, '.')
    if l_new_2 is not None:
        # Good, there is still some liberty remaining - but if it's
        # just the two, check that we are not caught in a ladder...
        # (Except that we don't care if we already have some alternative
        # escape routes!)
        if solutions or not (contact(board_put(fboard, l_new_2, 'L'), '.') is None
                             and read_ladder_attack(escpos, l, l_new, l_new_2) is not None):
            solutions.append(l)

    return (True, solutions)


def cfg_distances(board, c):
    """ return a board map listing common fate graph distances from
    a given point - this corresponds to the concept of locality while
    contracting groups to single points """
    cfg_map = W*W*[-1]
    cfg_map[c] = 0

    # flood-fill like mechanics
    fringe = [c]
    while fringe:
        c = fringe.pop()
        for d in neighbors(c):
            if board[d].isspace() or 0 <= cfg_map[d] <= cfg_map[c]:
                continue
            cfg_before = cfg_map[d]
            if board[d] != '.' and board[d] == board[c]:
                cfg_map[d] = cfg_map[c]
            else:
                cfg_map[d] = cfg_map[c] + 1
            if cfg_before < 0 or cfg_before > cfg_map[d]:
                fringe.append(d)
    return cfg_map


def line_height(c):
    """ Return the line number above nearest board edge """
    row, col = divmod(c - (W+1), W)
    return min(row, col, N-1-row, N-1-col)


def empty_area(board, c, dist=3):
    """ Check whether there are any stones in Manhattan distance up
    to dist """
    for d in neighbors(c):
        if board[d] in 'Xx':
            return False
        elif board[d] == '.' and dist > 1 and not empty_area(board, d, dist-1):
            return False
    return True


# 3x3 pattern routines (those patterns stored in pat3src above)

def pat3_expand(pat):
    """ All possible neighborhood configurations matching a given pattern;
    used just for a combinatoric explosion when loading them in an
    in-memory set. """
    def pat_rot90(p):
        return [p[2][0] + p[1][0] + p[0][0], p[2][1] + p[1][1] + p[0][1], p[2][2] + p[1][2] + p[0][2]]
    def pat_vertflip(p):
        return [p[2], p[1], p[0]]
    def pat_horizflip(p):
        return [l[::-1] for l in p]
    def pat_swapcolors(p):
        return [l.replace('X', 'Z').replace('x', 'z').replace('O', 'X').replace('o', 'x').replace('Z', 'O').replace('z', 'o') for l in p]
    def pat_wildexp(p, c, to):
        i = p.find(c)
        if i == -1:
            return [p]
        return reduce(lambda a, b: a + b, [pat_wildexp(p[:i] + t + p[i+1:], c, to) for t in to])
    def pat_wildcards(pat):
        return [p for p in pat_wildexp(pat, '?', list('.XO '))
                  for p in pat_wildexp(p, 'x', list('.O '))
                  for p in pat_wildexp(p, 'o', list('.X '))]
    return [p for p in [pat, pat_rot90(pat)]
              for p in [p, pat_vertflip(p)]
              for p in [p, pat_horizflip(p)]
              for p in [p, pat_swapcolors(p)]
              for p in pat_wildcards(''.join(p))]

pat3set = set([p.replace('O', 'x') for p in pat3src for p in pat3_expand(p)])

def neighborhood_33(board, c):
    """ return a string containing the 9 points forming 3x3 square around
    a certain move candidate """
    return (board[c-W-1 : c-W+2] + board[c-1 : c+2] + board[c+W-1 : c+W+2]).replace('\n', ' ')


# large-scale pattern routines (those patterns living in patterns.{spat,prob} files)

# are you curious how these patterns look in practice? get
# https://github.com/pasky/pachi/blob/master/tools/pattern_spatial_show.pl
# and try e.g. ./pattern_spatial_show.pl 71

spat_patterndict = dict()  # hash(neighborhood_gridcular()) -> spatial id
def load_spat_patterndict(f):
    """ load dictionary of positions, translating them to numeric ids """
    for line in f:
        # line: 71 6 ..X.X..OO.O..........#X...... 33408f5e 188e9d3e 2166befe aa8ac9e 127e583e 1282462e 5e3d7fe 51fc9ee
        if line.startswith('#'):
            continue
        neighborhood = line.split()[2].replace('#', ' ').replace('O', 'x')
        spat_patterndict[hash(neighborhood)] = int(line.split()[0])

large_patterns = dict()  # spatial id -> probability
def load_large_patterns(f):
    """ dictionary of numeric pattern ids, translating them to probabilities
    that a move matching such move will be played when it is available """
    # The pattern file contains other features like capture, selfatari too;
    # we ignore them for now
    for line in f:
        # line: 0.004 14 3842 (capture:17 border:0 s:784)
        p = float(line.split()[0])
        m = re.search('s:(\d+)', line)
        if m is not None:
            s = int(m.groups()[0])
            large_patterns[s] = p


def neighborhood_gridcular(board, c):
    """ Yield progressively wider-diameter gridcular board neighborhood
    stone configuration strings, in all possible rotations """
    # Each rotations element is (xyindex, xymultiplier)
    rotations = [((0,1),(1,1)), ((0,1),(-1,1)), ((0,1),(1,-1)), ((0,1),(-1,-1)),
                 ((1,0),(1,1)), ((1,0),(-1,1)), ((1,0),(1,-1)), ((1,0),(-1,-1))]
    neighborhood = ['' for i in range(len(rotations))]
    wboard = board.replace('\n', ' ')
    for dseq in pat_gridcular_seq:
        for ri in range(len(rotations)):
            r = rotations[ri]
            for o in dseq:
                y, x = divmod(c - (W+1), W)
                y += o[r[0][0]]*r[1][0]
                x += o[r[0][1]]*r[1][1]
                if y >= 0 and y < N and x >= 0 and x < N:
                    neighborhood[ri] += wboard[(y+1)*W + x+1]
                else:
                    neighborhood[ri] += ' '
            yield neighborhood[ri]


def large_pattern_probability(board, c):
    """ return probability of large-scale pattern at coordinate c.
    Multiple progressively wider patterns may match a single coordinate,
    we consider the largest one. """
    probability = None
    matched_len = 0
    non_matched_len = 0
    for n in neighborhood_gridcular(board, c):
        sp_i = spat_patterndict.get(hash(n))
        prob = large_patterns.get(sp_i) if sp_i is not None else None
        if prob is not None:
            probability = prob
            matched_len = len(n)
        elif matched_len < non_matched_len < len(n):
            # stop when we did not match any pattern with a certain
            # diameter - it ain't going to get any better!
            break
        else:
            non_matched_len = len(n)
    return probability


#######################
# board string routines

def neighbors(c):
    """ generator of coordinates for all neighbors of c """
    return [c-1, c+1, c-W, c+W]

def diag_neighbors(c):
    """ generator of coordinates for all diagonal neighbors of c """
    return [c-W-1, c-W+1, c+W-1, c+W+1]


def board_put(board, c, p):
    return board[:c] + p + board[c+1:]


def floodfill(board, c):
    """ replace continuous-color area starting at c with special color # """
    # This is called so much that a bytearray is worthwhile...
    byteboard = bytearray(board, 'utf-8')
    p = byteboard[c]
    byteboard[c] = ord('#')
    fringe = [c]
    while fringe:
        c = fringe.pop()
        for d in neighbors(c):
            if byteboard[d] == p:
                byteboard[d] = ord('#')
                fringe.append(d)
    return str(byteboard)


# Regex that matches various kind of points adjecent to '#' (floodfilled) points
contact_res = dict()
for p in ['.', 'x', 'X']:
    rp = '\\.' if p == '.' else p
    contact_res_src = ['#' + rp,  # p at right
                       rp + '#',  # p at left
                       '#' + '.'*(W-1) + rp,  # p below
                       rp + '.'*(W-1) + '#']  # p above
    contact_res[p] = re.compile('|'.join(contact_res_src), flags=re.DOTALL)

def contact(board, p):
    """ test if point of color p is adjecent to color # anywhere
    on the board; use in conjunction with floodfill for reachability """
    m = contact_res[p].search(board)
    if not m:
        return None
    return m.start() if m.group(0)[0] == p else m.end() - 1


def is_eyeish(board, c):
    """ test if c is inside a single-color diamond and return the diamond
    color or None; this could be an eye, but also a false one """
    eyecolor = None
    for d in neighbors(c):
        if board[d].isspace():
            continue
        if board[d] == '.':
            return None
        if eyecolor is None:
            eyecolor = board[d]
            othercolor = eyecolor.swapcase()
        elif board[d] == othercolor:
            return None
    return eyecolor

def is_eye(board, c):
    """ test if c is an eye and return its color or None """
    eyecolor = is_eyeish(board, c)
    if eyecolor is None:
        return None

    # Eye-like shape, but it could be a falsified eye
    falsecolor = eyecolor.swapcase()
    false_count = 0
    at_edge = False
    for d in diag_neighbors(c):
        if board[d].isspace():
            at_edge = True
        elif board[d] == falsecolor:
            false_count += 1
    if at_edge:
        false_count += 1
    if false_count >= 2:
        return None

    return eyecolor

class Position(namedtuple('Position', 'board cap n ko last last2 komi')):
    """ Implementation of simple Chinese Go rules;
    n is how many moves were played so far """

    def move(self, c):
        """ play as player X at the given coord c, return the new position """

        # Test for ko
        if c == self.ko:
            return None
        # Are we trying to play in enemy's eye?
        in_enemy_eye = is_eyeish(self.board, c) == 'x'

        board = board_put(self.board, c, 'X')
        # Test for captures, and track ko
        capX = self.cap[0]
        singlecaps = []
        for d in neighbors(c):
            if board[d] != 'x':
                continue
            # XXX: The following is an extremely naive and SLOW approach
            # at things - to do it properly, we should maintain some per-group
            # data structures tracking liberties.
            fboard = floodfill(board, d)  # get a board with the adjecent group replaced by '#'
            if contact(fboard, '.') is not None:
                continue  # some liberties left
            # no liberties left for this group, remove the stones!
            capcount = fboard.count('#')
            if capcount == 1:
                singlecaps.append(d)
            capX += capcount
            board = fboard.replace('#', '.')  # capture the group
        # Set ko
        ko = singlecaps[0] if in_enemy_eye and len(singlecaps) == 1 else None
        # Test for suicide
        if contact(floodfill(board, c), '.') is None:
            return None

        # Update the position and return
        return Position(board=board.swapcase(), cap=(self.cap[1], capX),
                        n=self.n + 1, ko=ko, last=c, last2=self.last, komi=self.komi)

    def pass_move(self):
        """ pass - i.e. return simply a flipped position """
        return Position(board=self.board.swapcase(), cap=(self.cap[1], self.cap[0]),
                        n=self.n + 1, ko=None, last=None, last2=self.last, komi=self.komi)

    def moves(self, i0):
        """ Generate a list of moves (includes false positives - suicide moves;
        does not include true-eye-filling moves), starting from a given board
        index (that can be used for randomization) """
        i = i0-1
        passes = 0
        while True:
            i = self.board.find('.', i+1)
            if passes > 0 and (i == -1 or i >= i0):
                break  # we have looked through the whole board
            elif i == -1:
                i = 0
                passes += 1
                continue  # go back and start from the beginning
            # Test for to-play player's one-point eye
            if is_eye(self.board, i) == 'X':
                continue
            yield i

    def last_moves_neighbors(self):
        """ generate a randomly shuffled list of points including and
        surrounding the last two moves (but with the last move having
        priority) """
        clist = []
        for c in self.last, self.last2:
            if c is None:  continue
            dlist = [c] + list(neighbors(c) + diag_neighbors(c))
            random.shuffle(dlist)
            clist += [d for d in dlist if d not in clist]
        return clist

    def score(self, owner_map=None):
        """ compute score for to-play player; this assumes a final position
        with all dead stones captured; if owner_map is passed, it is assumed
        to be an array of statistics with average owner at the end of the game
        (+1 black, -1 white) """
        board = self.board
        i = 0
        while True:
            i = self.board.find('.', i+1)
            if i == -1:
                break
            fboard = floodfill(board, i)
            # fboard is board with some continuous area of empty space replaced by #
            touches_X = contact(fboard, 'X') is not None
            touches_x = contact(fboard, 'x') is not None
            if touches_X and not touches_x:
                board = fboard.replace('#', 'X')
            elif touches_x and not touches_X:
                board = fboard.replace('#', 'x')
            else:
                board = fboard.replace('#', ':')  # seki, rare
            # now that area is replaced either by X, x or :
        komi = self.komi if self.n % 2 == 1 else -self.komi
        if owner_map is not None:
            for c in range(W*W):
                n = 1 if board[c] == 'X' else -1 if board[c] == 'x' else 0
                owner_map[c] += n * (1 if self.n % 2 == 0 else -1)
        return board.count('X') - board.count('x') + komi


def empty_position():
    """ Return an initial board position """
    return Position(board=empty, cap=(0, 0), n=0, ko=None, last=None, last2=None, komi=7.5)

