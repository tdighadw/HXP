
from gymnasium import Env
from gymnasium.spaces import Discrete
from copy import deepcopy

# Following function is inspired by
# - https://stackabuse.com/how-to-print-colored-text-in-python/
# - https://github.com/openai/gym/blob/master/gym/utils/colorize.py

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

#  Highlight the background of text.
#  Input: text which need a colored background (str), a color (str) and use or not of different colors (bool)
#  Output: Text with colored background (str)
def colorize(text, color_, small=True):
    if small:
        num = color2num[color_]
        return (f"\x1b[{num}m{text}\x1b[0m")
    else:
        num = color_
        return (f"\x1b[48;5;{num}m{text}\x1b[0m")

#  Default game config, can be overridden in `env_config`
#  https://github.com/davidcotton/gym-connect4/blob/master/gym_connect4/envs/connect4_env.py
BOARD_HEIGHT = 6
BOARD_WIDTH = 7
WIN_LENGTH = 4
REWARD_WIN = 1.0
REWARD_LOSE = -1.0
REWARD_DRAW = 0.0
REWARD_STEP = 0.0

class Connect4(Env):

    def __init__(self, height=BOARD_HEIGHT, width=BOARD_WIDTH, win_length=WIN_LENGTH, rewards=[REWARD_WIN, REWARD_LOSE, REWARD_DRAW, REWARD_STEP]):
        #  Board initialised
        self.board = [[0 for _ in range(width)] for _ in range(height)]
        self.rows, self.cols = height, width
        #  Action space
        self.action_space = Discrete(width)
        #  Win length
        self.win_length = win_length
        #  Rewards
        self.rewards = {'win': rewards[0], 'lose': rewards[1], 'draw': rewards[2], 'step': rewards[3]}
        #  Window length for *minimax* algorithm
        self.window_length = 4

    #  Reset environment
    #  Input: None
    #  Output: None
    def reset(self):
        # Initialize the board
        self.board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        return self.board

    #  Perform action of player 1, then simulate the player 2 turn (update the board)
    #  Input: agents (Agent list), action from player 1 (int), partial Experience of player 2 (list), NN (DQN),
    #  epsilon (float), learning agent (str), device type (str), use of deterministic transition(s) (bool)
    #  Output: reward (float), done (bool), new state (int list) for player 1 and player 2 Experience and partial
    #  Experience (int list)
    def step(self, agents, action, p2_exp=[], net=None, epsilon=0.0, learning_agent='', device="cpu", det_transition=False):
        p1, p2 = agents
        #  Update the board
        self.update(p1, action)
        #  Choose transition with p2 action
        p2_action = p2.predict(self.inverse_board(self.board), net, epsilon, device)
        #  Complete p2 experience
        if learning_agent == p2.id and not det_transition:
            if p2_exp != []:
                p2_reward, p2_done = self.reward_done_function(p2)
                p2_exp += [p2_reward, p2_done, deepcopy(self.board)]

            p2_partial_exp = [deepcopy(self.board), p2_action]  # Allow the construction of future experience

        #  Update the board using p2 action (transition)
        if p2_action is not None:  # last action is None when p1 has already won the game)
            self.update(p2, p2_action)
        #  Used for approximate HXp
        if det_transition:
            return self.board
        else:
            #  Get reward and done for agent
            reward, done = self.reward_done_function(p1)
            # self.board is a new state for p1 (updated inside reward_done_function)
            if learning_agent == p2.id:
                return reward, done, self.board, p2_exp, p2_partial_exp
            else:
                return reward, done, self.board, p2_exp, []

    #  At a timestep, compute the reward for a specific agent and determine if the board configuration is terminal
    #  Input: agent (Agent)
    #  Output: reward signal (float) and done (bool)
    def reward_done_function(self, agent):
        #  Reward is 0 by default
        reward = self.rewards['step']
        done = False
        #  Check win/lose/draw conditions
        if self.win(agent.token):  # Win
            reward = self.rewards['win']
            done = True
        elif self.win(agent.token*(-1)):  # Lose
            reward = self.rewards['lose']
            done = True
        elif not sum(line.count(0) for line in self.board):  # Draw
            reward = self.rewards['draw']
            done = True

        return reward, done

    #  Update board
    #  Input: agent (Agent), action (int)
    #  Output: None
    def update(self, agent, action):
        #  Get the location to add a token
        column = [row[action] for row in self.board]
        #  Finding the last occurence of 0, i.e. the line to put the token
        line_index = max(idx for idx, elm in enumerate(column) if elm == 0)
        #  Update the board
        self.board[line_index][action] = agent.token
        return

    #  Update a specific board which is not the current one depicted by the environment
    #  Input: specific board (int list list), token (int) and action (int)
    #  Output: updated specific board (int list list)
    def update_state(self, state, token, action):
        #print('update_state: \n state: {} \n token: {} \n action: {}'.format(state, token, action))

        #  Get the location to add a token
        column = [row[action] for row in state]
        #  Finding the last occurence of 0, i.e. the line to put the token
        line_index = max(idx for idx, elm in enumerate(column) if elm == 0)
        #  Update the board
        state[line_index][action] = token

        return state

    #  Check win/lose/draw conditions and return the corresponding reward
    #  Source: http://romain.raveaux.free.fr/document/ReinforcementLearningbyQLearningConnectFourGame.html
    #  Input: token of the agent (int) and optional board (int list list)
    #  Output: reward signal (float)
    def win(self, token, copy_board=None, hxp=False):
        board = self.board if copy_board is None else copy_board
        cols = self.cols
        rows = self.rows
        if hxp: opp_token = token * (-1)

        # Check horizontal locations for win
        for c in range(cols - 3):
            for r in range(rows):
                if board[r][c] == token and board[r][c + 1] == token and board[r][c + 2] == token and board[r][c + 3] == token:
                    return True
                if hxp and board[r][c] == opp_token and board[r][c + 1] == opp_token and board[r][c + 2] == opp_token and board[r][c + 3] == opp_token:
                    return True


        # Check vertical locations for win
        for c in range(cols):
            for r in range(rows - 3):
                if board[r][c] == token and board[r + 1][c] == token and board[r + 2][c] == token and board[r + 3][c] == token:
                    return True
                if hxp and board[r][c] == opp_token and board[r + 1][c] == opp_token and board[r + 2][c] == opp_token and board[r + 3][c] == opp_token:
                    return True


        for c in range(cols - 3):
            # Check positively sloped diagonals
            for r in range(rows - 3):
                if board[r][c] == token and board[r + 1][c + 1] == token and board[r + 2][c + 2] == token and board[r + 3][c + 3] == token:
                    return True
                if hxp and board[r][c] == opp_token and board[r + 1][c + 1] == opp_token and board[r + 2][c + 2] == opp_token and board[r + 3][c + 3] == opp_token:
                    return True

            # Check negatively sloped diagonals
            for r in range(3, rows):
                if board[r][c] == token and board[r - 1][c + 1] == token and board[r - 2][c + 2] == token and board[r - 3][c + 3] == token:
                    return True
                if hxp and board[r][c] == opp_token and board[r - 1][c + 1] == opp_token and board[r - 2][c + 2] == opp_token and board[r - 3][c + 3] == opp_token:
                    return True

        return False

    #  Display to the user the current state of the environment
    #  Input: None
    #  Output: None
    def render(self):
        #  Print the top of the board
        print(' '.join((self.cols+1) * '-'))
        #  Print tokens
        for i in range(self.rows):
            str_tmp = '|'
            for j in range(self.cols):
                if self.board[i][j] != 0:
                    if self.board[i][j] is None:
                        color = "white"
                    else:
                        color = "red" if self.board[i][j] == -1 else "yellow"
                    str_tmp += colorize(' ', color)
                else:
                    str_tmp += ' '
                str_tmp += '|'
            print(str_tmp)

        #  Print the bottom of the board
        print(' '.join((self.cols+1) * '-'))
        print()
        return

    #  Inverse tokens in the board
    #  Input: board (int list list)
    #  Output: reversed board (int list list)
    def inverse_board(self, board):
        new_board = deepcopy(board)

        for i in range(self.rows):
            for j in range(self.cols):
                if board[i][j] == 1:
                    new_board[i][j] = -1
                elif board[i][j] == -1:
                    new_board[i][j] = 1

        return new_board

    #  Get the action space
    #  Input : None
    #  Output : action space (Discrete)
    def get_action_space(self):
        return self.action_space
