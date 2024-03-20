#  WARNING : this version of the environment only works when there is only one starting state S in the map
#  Need updates to allow several starting states
#  This Frozen Lake environment is inspired by: https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
import random

from gymnasium import Env
from gymnasium import utils
from gymnasium.spaces import Discrete
import numpy as np

#  Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

#  Set of maps
MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],

    "6x6": [
        "SFFFFF",
        "FFHFFF",
        "FFHFFF",
        "FFFHHF",
        "FFFFFF",
        "HFFFFG"
    ],

    "7x7": [
        "SFFFFFH",
        "FFFFFFH",
        "FHHFFFH",
        "FHHFFFH",
        "FFFFFFF",
        "FFFFFFF",
        "HHHHFFG"
    ],

    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],

    "10x10": [
        "HFHFFHHGHH",
        "FFFHFFFFFF",
        "FHFFFFFHHF",
        "HFFFHHFFFF",
        "FFHFHFHFFH",
        "FFHFFFFFHH",
        "FHFFFHHFFH",
        "SFFFFFFHFF",
        "FFFHHFFFFH",
        "HFFFFFHFHH"
    ],

    "corridor13": [
        "FFHFFHHFFHFHF",
        "SFFFFFFFFFFFG",
        "FFHFFHHFFHFHF"
    ],

    "corridor10": [
        "FFHFFHFFHF",
        "SFFFFFFFFG",
        "FFHFFHFFHF"
    ],

    "corridor8": [
        "FHFFHHFF",
        "SFFFFFFG",
        "FHFFHHFF"
    ],

    "corridor6": [
        "FFHFHF",
        "SFFFFG",
        "FFHFHF"
    ],


}

#  Class used for coloring the current state in the render function
class bcolors:
    OKCYAN = '\033[96m'
    ENDC = '\033[0m'

#  Compute the Manhattan distance
#  Input: two x,y coordinates (int list)
#  Output: Manhattan distance (int)
def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

#  Sort points according to the Manhattan distance
#  Input: points (int list list), x,y coordinates (int)
#  Output: sorted points (int list list)
def get_ordered_list(points, x, y):
    points.sort(key=lambda p: manhattan_dist(p, [x, y]))
    return points

class MyFrozenLake(Env):

    def __init__(self, map_name="4x4", is_slippery=True, slip_probas=[1/3, 1/3, 1/3], many_features=False):
        #  Map
        desc = MAPS[map_name]
        self.desc = np.asarray(desc, dtype="c")
        #  Action space
        self.action_space = Discrete(4)
        #  Dimension of the map
        self.nRow, self.nCol = self.desc.shape
        #  Define the type of state
        self.many_features = many_features
        # Number of Actions, States
        nA = 4
        nS = self.nRow*self.nCol
        # Last action (useful for the render)
        self.lastaction = None
        # Probabilities to slip
        self.slip_probas = slip_probas
        #  Probability matrix for the agent
        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)} if not self.many_features else {}
        #  Get the goal position (only one goal allowed)
        row_goal, col_goal = None, None
        #  Total number of holes and starting cell position (used if self.many_features)
        self.hole_cpt = 0
        #  Start position
        self.start_position = None
        for i in range(len(self.desc)):
            for j in range(len(self.desc[0])):
                if bytes(self.desc[i, j]) in b"G":
                    row_goal, col_goal = i, j
                if self.many_features:
                    if bytes(self.desc[i, j]) in b"H":
                        self.hole_cpt += 1
                    elif bytes(self.desc[i, j]) in b"S":
                        self.start_position = [i, j]
        #  Goal position
        self.goal_position = [row_goal, col_goal]
        #  Initial state
        self.state = self.init_state()

        #  Store a transition
        #  Input: coordinates (int), action (int), transition probability (float), transitions storage (transition list)
        #  Output: updated transitions storage (transition list)
        def tr(row, col, action, proba, li):
            newrow, newcol = self.inc(row, col, action)
            newletter = self.desc[newrow, newcol]
            done = bytes(newletter) in b"GH"
            reward = float(newletter == b"G")
            # state is composed of one feature
            if not self.many_features:
                newstate = self.to_s(newrow, newcol)
                li.append(
                    (proba, newstate, reward, done))

            # state is composed of several features
            else:
                holes = self.close_hole(newrow, newcol, one_hole=False)
                # print(holes)

                if len(holes) == 1:
                    newstate = self.to_s(newrow, newcol, row, col)
                    li.append((proba, newstate, reward, done))

                else:
                    for i in range(len(holes)):
                        newstate = self.to_s(newrow, newcol, row, col, i)
                        li.append((proba / len(holes), newstate, reward, done))
            return li

        #  Get the set of possible previous positions of the agent located at row,col coordinates
        #  Input: x,y coordinates (int)
        #  Output: list of possible previous agent's positions (int list list)
        def last_positions(row, col):
            last_positions = []
            #  Wall detection (no movement)
            if not row or row == self.nRow - 1 or not col or col == self.nCol - 1:
                last_positions.append(self.to_position(row, col))

            #  'Right' transition
            if col - 1 >= 0 and self.desc[row, col - 1] not in b"GH":
                last_positions.append(self.to_position(row, col - 1))

            # 'Left' transition
            if col + 1 < self.nCol and self.desc[row, col + 1] not in b"GH":
                last_positions.append(self.to_position(row, col + 1))

            # 'Up' transition
            if row - 1 >= 0 and self.desc[row - 1, col] not in b"GH":
                last_positions.append(self.to_position(row - 1, col))

            # 'Down' transition
            if row + 1 < self.nRow and self.desc[row + 1, col] not in b"GH":
                last_positions.append(self.to_position(row + 1, col))

            return last_positions

        #  Update the probability matrix
        #  Input: x,y coordinates (int), state (int (list)), slippery surface (bool)
        #  Output: None
        def update(row, col, s, is_slippery):
            for a in range(4):
                li = []
                letter = self.desc[row, col]
                if letter in b"GH":
                    li.append((1.0, s, 0, True))
                else:
                    if is_slippery:
                        i = 0
                        for b in [(a - 1) % 4, a, (a + 1) % 4]:
                            li = tr(row, col, b, self.slip_probas[i], li)
                            i += 1
                    else:
                        li = tr(row, col, a, 1.0, li)
                self.P[s][a] = li

        # Fill the probability matrix
        for row in range(self.nRow):
            for col in range(self.nCol):
                # state is composed of one feature
                if not self.many_features:
                    s = self.to_s(row, col)
                    update(row, col, s, is_slippery)

                # state is composed of several features
                else:
                    for last_pos in last_positions(row, col):
                        for hole in self.close_hole(row, col, one_hole=False):
                            s = self.to_position(row, col), last_pos, hole , manhattan_dist(self.start_position, [row, col]), self.hole_cpt
                            self.P[s] = {}
                            update(row, col, s, is_slippery)

        #print(self.P.keys())
        return

    #  From x,y coordinates to position
    #  Input: x,y coordinates (int)
    #  Output: position (int)
    def to_position(self, row, col):
        return row * self.nCol + col

    #  From coordinates to state
    #  Input: coordinates (int)
    #  Output: a state (int)
    def to_s(self, row, col, last_row=None, last_col=None, hole_idx=None):
        agent_position = self.to_position(row, col)
        # state composed of one feature (also useful for predicate check)
        if not self.many_features or (last_row is None and last_col is None):
            return agent_position

        # state composed of several features
        else:
            if hole_idx is not None:
                hole = self.close_hole(row, col, one_hole=False)[hole_idx]
            else:
                hole = self.close_hole(row, col)

            return agent_position, self.to_position(last_row, last_col), hole, manhattan_dist(self.start_position, [row, col]), self.hole_cpt

    #  Update coordinates
    #  Input: coordinates (int), action (int)
    #  Output: updated coordinates (int)
    def inc(self, row, col, a):
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, self.nRow - 1)
        elif a == RIGHT:
            col = min(col + 1, self.nCol - 1)
        elif a == UP:
            row = max(row - 1, 0)
        return (row, col)

    #  The agent performs a step in the environment and arrives in a new state
    #  according to the probability matrix, the current state, the action of both the agent and the wind
    #  Input: action (int), index of the action (int), already chosen new state (int)
    #  Output: new state (int (list)), reward (int), terminal state (bool), probability to reach the new state (float)
    def step(self, action, new_state=None):
        #  Case of the agent's step
        transitions = self.P[self.state][action]
        #print(transitions)

        #  Random choice among possible transitions
        i = np.random.choice(len(transitions), p=[t[0] for t in transitions])
        p, s, r, d = transitions[i]

        #  Updates
        self.state = s
        self.lastaction = action
        return s, r, d, p

    #  Reset the environment
    #  Input: None
    #  Output: initial state (int (list))
    def reset(self):
        self.state = self.init_state()
        self.lastaction = None
        return self.state

    #  Render the environment
    #  Input:  None
    #  Output: None
    def render(self):

        #  Print the current action
        if self.lastaction != None:
            print("    ({})".format(["Left", "Down", "Right", "Up"][self.lastaction]))

        #  Get the current position
        idx_position = self.state if not self.many_features else self.state[0]
        row, col = idx_position // self.nCol, idx_position % self.nCol
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]

        #  Highlight current position in red
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        #  Render
        for line in range(self.nRow):
            row_str = ""
            for column in range(self.nCol):
                row_str = row_str + desc[line][column]
            print(row_str)

        return

    #  Set a state and last action
    #  Input: state (int (list))
    #  Output: None
    def setObs(self, obs):
        self.state = obs
        self.lastaction = None
        return

    #  Initialize the agent's state
    #  Input: None
    #  Output: initial state (int (list))
    def init_state(self):
        for i in range(len(self.desc)):
            for j in range(len(self.desc[0])):
                if bytes(self.desc[i, j]) in b"S":
                    if not self.many_features:
                        return self.to_s(i, j)
                    else:
                        return self.to_s(i, j, i, j)

    # Get the position(s) of one (all) of the close hole(s)
    # Input: coordinates (int) and number of position to return (bool)
    # Output: position(s) (int list)
    def close_hole(self, row, col, one_hole=True):
        #  Early stop (agent is already in a hole
        if bytes(self.desc[row, col]) in b"H":
            hole = self.to_position(row, col)
            return hole if one_hole else [hole]
        holes = []
        view_range = self.nCol - 1
        n = 2

        #  Get holes
        view = self.neighbor(row, col, view_range)
        #print('view: {}'.format(view))

        # Extract holes
        for (i,j) in view:
            #  Add hole position
            if bytes(self.desc[i, j]) in b"H" and len(holes) != n:
                #print('hole! {}'.format(self.to_position(i, j)))
                holes.append(self.to_position(i, j))

        if one_hole:
            h = random.choice(holes)
            return h
        else:
            return holes[0] if len(holes) == 1 else holes

    #  Get the neighborhood located at exactly view_range distance from the agent
    #  Input: coordinates (int), view range (int)
    #  Output: list of map indexes (list)
    def neighbor(self, row, col, view_range):
        # col idxs
        upper_col_idx = min(col + view_range, self.nCol - 1)
        lower_col_idx = max(col - view_range, 0)
        # row idxs
        upper_row_idx = min(row + view_range, self.nRow - 1)
        lower_row_idx = max(row - view_range, 0)

        L = [(l,c) for c in range(lower_col_idx, upper_col_idx + 1) for l in range(lower_row_idx, upper_row_idx + 1)]

        tmp_list = get_ordered_list(L, row, col)
        indexes = []
        for (l, c) in tmp_list:
            indexes.append((l, c))

        return indexes
