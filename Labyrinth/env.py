from copy import deepcopy

#  Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

#  Set of maps
LABYRINTHS = {
    "4x4": [['-','-','S','-'],
            ['-','P','.','-'],
            ['-','-','.','-'],
            ['-','E','.','-']],

    "8x8": [['-', '-', '-', '-', '-', '-', '-', '-'],
            ['-', '.', '.', '.', '.', 'P', '-', '-'],
            ['-', '.', '-', '-', '.', '-', '.', '-'],
            ['E', '.', '.', '.', '-', '.', '.', '-'],
            ['-', 'P', '-', '.', '.', '.', 'P', '-'],
            ['-', '-', '-', '.', '-', '-', '-', '-'],
            ['-', 'P', '.', '.', '.', '.', '.', '-'],
            ['-', '-', '-', '-', '-', '-', 'S', '-']],

    "corridor": [['.', '.', '.', '.', '.', '.', '.', '.'],
                 ['E', '-', '-', '-', '-', '-', '-', '-'],
                 ['.', '.', '.', '.', '.', '.', '.', 'S']],

    "crossroad": [['-', '-', '-', 'S', '-', '-', '-'],
                ['S', '-', '-', '.', '-', '-', '-'],
                ['.', '-', '.', '.', '-', '-', '-'],
                ['.', 'P', '.', 'P', '.', '.', 'S'],
                ['.', '.', 'E', '.', '.', '-', '-'],
                ['-', 'P', '.', 'P', '-', '-', '-'],
                ['-', '-', '.', '.', '.', 'S', '-']]
    
}

#  Dict of colors
color2num = dict(
    gray=40,
    red=41,
    green=42,
    yellow=43,
    blue=44,
    magenta=45,
    cyan=46,
    white=47,
)

#  Display current agent's position by highlighting the background
#  Input: text to highlight (str), color (str), change highlight style (bool)
#  Output: highlighted text (str)
def colorize(text, color_, small=True):
    if small:
        num = color2num[color_]
        return (f"\x1b[{num}m{text}\x1b[0m")
    else:
        num = color_
        return (f"\x1b[48;5;{num}m{text}\x1b[0m")


class LabrinthEnv:

    def __init__(self, map_name='4x4', proba=[]):
        # Map
        self.map = LABYRINTHS[map_name]
        # Reachable states
        self.states = self.get_states()
        # Action space
        self.actions = 4
        # Transition matrix
        self.P = {s: {a: [] for a in range(self.actions)} for s in self.states}
        # Current state and last action
        self.s = self.init_state()
        self.lastaction = None
        # Proba in case of stochastic environment ([x, y] agent has x% to stay in its case, y% to move in the current direction)
        self.proba = proba

        #  Reward function
        #  Input: type of state (str)
        #  Output: reward (int)
        def reward_function(str_state):
            if str_state == 'S':
                return 1
            else:
                return 0

        #  Update the probability matrix
        #  Input: current x,y coordinates (int), action to perform (int), agent's move (bool)
        #  Output: new state reached (int), obtained reward (int), terminal state (bool)
        def update_probability_matrix(row, col, action, no_move=False):
            # stay in its current cell
            if no_move:
                s = self.from_coord_to_s(row, col)
                return s, 0, False
            # move to another cell
            else:
                #  Coordinates of new state
                new_row, new_col = self.inc(row, col, action)
                #  New state
                str_new_state = self.map[new_row][new_col]
                new_state = self.from_coord_to_s(new_row, new_col)
                #  Reward
                reward = reward_function(str_new_state)
                #  Done
                done = str_new_state in ['S', 'P']

                return new_state, reward, done

        #  Fill the probability matrix
        for row in range(len(self.map)):
            for col in range(len(self.map[0])):
                str_s = self.map[row][col]
                s = self.from_coord_to_s(row, col)
                # Do not create transitions from walls!
                if s in self.states:
                    for a in range(self.actions):
                        tr = self.P[s][a]
                        # From a terminal state, there is no transition
                        if str_s in ['S', 'P']:
                            tr.append((1.0, s, 0, True))
                        else:
                            # stocha. env.
                            if self.proba:
                                for idx, p in enumerate(self.proba):
                                    no_move = not idx
                                    tr.append((p, *update_probability_matrix(row, col, a, no_move)))
                            # det. env.
                            else:
                                tr.append((1.0, *update_probability_matrix(row, col, a)))

        print('self.states: {}'.format(self.states))
        print('self.actions: {}'.format(self.actions))
        print('self.P: {}'.format(self.P))
        print('self.s: {}'.format(self.s))
        print('self.lastaction: {}'.format(self.lastaction))
        return

    #  From x,y coordinates to state
    #  Input: x,y coordinates (int)
    #  Output: state (int)
    def from_coord_to_s(self, row, col):
        return row * len(self.map[0]) + col

    #  Get the initial state
    #  Input: None
    #  Output: initial state (int)
    def init_state(self):
        for row in range(len(self.map)):
            for col in range(len(self.map[0])):
                if self.map[row][col] == 'E':
                    return self.from_coord_to_s(row, col)
        return -1

    #  Modify x,y coordinates according to an action a
    #  Input: x,y coordinates (int), action (int)
    #  Output: updated x,y coordinates (int)
    def inc(self, row, col, a):
        if a == LEFT:
            if self.map[row][max(col-1, 0)] != '-':
                col = max(col - 1, 0)
        elif a == DOWN:
            if self.map[min(row + 1, len(self.map) - 1)][col] != '-':
                row = min(row + 1, len(self.map) - 1)
        elif a == RIGHT:
            if self.map[row][min(col + 1, len(self.map[0]) - 1)] != '-':
                col = min(col + 1, len(self.map[0]) - 1)
        elif a == UP:
            if self.map[max(row - 1, 0)][col] != '-':
                row = max(row - 1, 0)
        return (row, col)

    #  Get the state space
    #  Input: None
    #  Output: list of states (int list)
    def get_states(self):
        states = []
        for row in range(len(self.map)):
            for col in range(len(self.map[0])):
                if self.map[row][col] != '-':
                    states.append(self.from_coord_to_s(row, col))
        return states

    #  Render the Labyrinth
    #  Input: None
    #  Output: None
    def render(self):
        #  Print the last action
        if self.lastaction != None:
            print("    ({})".format(["Left", "Down", "Right", "Up"][self.lastaction]))

        #  Get the current position depending on the agent's type
        row, col = self.s // len(self.map[0]), self.s % len(self.map[0])
        #  Highlight current position in red
        map_copy = deepcopy(self.map)
        map_copy[row][col] = colorize(map_copy[row][col], "blue")

        #  Render
        for line in range(len(map_copy)):
            row_str = ""
            for column in range(len(map_copy[0])):
                row_str = row_str + map_copy[line][column]
            print(row_str)

        return

    #  Perform a step in the Labyrinth the initial state
    #  Input: action (int)
    #  Output: new state (int), reward (int), terminal state (bool),
    #  probability to arrive in new_state by doing action from self.s (float), None
    def step(self, action):
        self.lastaction = action
        transitions = self.P[self.s][self.lastaction]
        if len(transitions) == 1:
            prob, new_state, reward, done = transitions[0]
            self.s = new_state
            return new_state, reward, done, prob, None
        else:
            return

    #  Reset the Labyrinth
    #  Input: None
    #  Output: initial state (int), None
    def reset(self):
        self.s = self.init_state()
        self.lastaction = None
        return self.s, None

    #  Set the current state and last action
    #  Input: state (int)
    #  Output: None
    def set_obs(self, obs):
        self.s = obs
        self.lastaction = None
        return
