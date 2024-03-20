import numpy as np
import json
import os

#  Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

#  Compute the argmax of an array
#  Input: an array (list)
#  Output: index of the maximum value (int)
def argmax(array):
    array = list(array)
    return array.index(max(array))

#  Build a list of decreasing values of exploratory rate
#  Input: number of episodes (int), minimum exploratory rate (float)
#  Output: list of exploratory rate (np.array)
def exprate_schedule(nE, exp_rate_min=0.05):
    x = np.arange(nE) + 1
    exp_rate_decay = exp_rate_min**(1 / nE)
    y = [max((exp_rate_decay**x[i]), exp_rate_min) for i in range(len(x))]
    return y

class Agent:

    def __init__(self, name, env, lr=.2, exp_rate=1.0, decay_gamma=0.95):
        #  Version
        self.name = name
        #  Parameters for updating Q
        self.lr = lr
        self.exp_rate = exp_rate
        self.decay_gamma = decay_gamma
        #  Environment
        self.env = env
        #  Actions
        self.actions = 4
        #  States
        self.states = list(env.P.keys())
        #  Q table
        self.Q = {s: [0.0 for _ in range(self.actions)] for s in self.states}

    #  Train an agent
    #  Input: number of training episode (int)
    #  Output: None
    def train(self, nE):
        #  Set list of exploratory rates
        expRate_schedule = exprate_schedule(nE)
        #  Training loop
        for episode in range(nE):

            #  Display the training progression
            if episode % 500 == 0:
                print("Episodes : " + str(episode) + ' - Exp rate : ' + str(expRate_schedule[episode]))

            #  Flag to stop the episode
            done = False
            while not done:

                #  Useful to keep for updating Q
                current_state = self.env.state
                #  Choose an action
                self.exp_rate = expRate_schedule[episode]
                action = self.chooseAction(current_state)
                #  Execute it
                new_state, reward, done, _ = self.env.step(action)
                #  Update Q value
                self.updateQ(current_state, action, new_state, reward)

            #  Reset environment
            self.env.reset()


        print("End of training")
        return

    #  Choose an action to perform from a state
    #  Input: current state (int (list))
    #  Output: action (int)
    def chooseAction(self, state):
        #  Exploratory move
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.randint(0, self.actions)
            return action
        #  Greedy move
        else:
            action = argmax(self.Q[state])
            return action

    #  Choose the new state for a wind agent
    #  Input: current state (int (list)) and an action (int)
    #  Output: new state (int)
    def chooseNewState(self, state, action):
        #  Perform action
        env = self.env
        new_row, new_col = env.inc((state // self.actions) // env.nCol, (state // self.actions) % env.nCol, action)
        new_state = env.to_s(new_row, new_col)
        #  Set list of reachable states
        reachable_state_list = [new_state * self.actions + i for i in range(self.actions)]
        #  Select one state using Agent's Q-table
        index_new_state = np.argmax(self.Q[new_state])
        return reachable_state_list[index_new_state]

    #  Update a value in the Q table
    #  Input: state (int (list)), action (int), reached state (int (list)), reward obtained (float)
    #  Output:  None
    def updateQ(self, state, action, new_state, reward):
        #print(self.Q[state])

        # Updating rule
        self.Q[state][action] = self.Q[state][action] + self.lr * \
                                (reward + self.decay_gamma * max(self.Q[new_state]) - self.Q[state][action])

        return

    #  Save the current Q table in a JSON file
    #  Input: directory path (str)
    #  Output: None
    def save(self, path):
        q_function_list = list(self.Q.items())
        with open(path + os.sep + 'Q_' + self.name, 'w') as fp:
            json.dump(q_function_list, fp)

        return

    #  Load a Q table from a JSON file
    #  Input: directory path (str)
    #  Output: None
    def load(self, path):
        absolute_dir_path = os.path.dirname(__file__)
        with open(absolute_dir_path + os.sep + path + os.sep + 'Q_' + self.name, 'r') as fp:
            q_list = json.load(fp)
            #print(q_list)
            # agent's state is defined with one feature (previous code version)
            if isinstance(q_list[0][0], float):
                self.Q = {s: q_values for s, q_values in enumerate(q_list)}
            # agent's state is defined with one feature (current code version)
            elif isinstance(q_list[0][0], int):
                self.Q = {s: q_values for s, q_values in q_list}
            # agent's state is defined with several features
            else:
                self.Q = {tuple(s): q_values for s, q_values in q_list}
            print("Q function loaded")

        return

    #  Predict an action from a given state
    #  Input: state (int (list))
    #  Output: state (int)
    def predict(self, observation):
        return argmax(self.Q[observation])
