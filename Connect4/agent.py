
import numpy as np
import torch

class Agent:

    def __init__(self, id, env, random=False):
        self.id = id
        self.env = env
        self.state = None
        self.token = 1 if self.id == 'Yellow' else -1
        self.actions = self.env.get_action_space().n
        self.random = random  # if True, the agent plays randomly

        return

    #  Choose the action to perform using the epsilon rate and DQN
    #  Input: state (int list list), NN (DQN), epsilon rate (float), device type (str)
    #  Output: action (int)
    def predict(self, state, net=None, epsilon=0.0, device="cpu"):
        #  Illegal actions
        illegal_actions = [i for i in range(self.env.action_space.n) if state[0][i] != 0]
        #  Exploratory move
        if self.random or np.random.random() < epsilon:
            action = self.random_action(illegal_actions)
        #  Greedy move
        else:
            action = self.policy(state, illegal_actions, net, device)
        return action

    #  Get the action to perform from a state using the already learnt DQN
    #  Input: state (int list list), set of illegal actions (int list), NN (DQN), device type (str)
    #  Output: action (int)
    def policy(self, state, illegal_actions, net, device='cpu'):
        #  Sort index based on q values from max to min
        max_idx_list = self.sort_actions(state, net, device)
        #  Predict the agent's action
        while max_idx_list[0] in illegal_actions:
            max_idx_list.pop(0)

        return max_idx_list[0]

    #  Sort actions according to state-action Q values
    #  Input: state (int list list), NN (DQN), device type (str)
    #  Output: sorted list of actions (int list)
    def sort_actions(self, state, net, device='cpu'):
        #  Convert into tensor
        tens_state = torch.tensor(state, dtype=torch.float32, device=device)
        tens_state = torch.unsqueeze(tens_state, dim=0)
        #  Get best action according to DQN
        with torch.no_grad():
            q_vals_v = net(tens_state)
        q_vals_list = q_vals_v.tolist()[0]

        return sorted(range(len(q_vals_list)), key=lambda k: q_vals_list[k], reverse=True)

    #  Provide to the agent a randomly chosen action
    #  Input: set of illegal actions (int list)
    #  Output: an action (int)
    def random_action(self, illegal_actions):
        action = self.env.action_space.sample()
        while (action in illegal_actions):
            action = self.env.action_space.sample()
        return action

# Method(s) to handle the exploration/exploitation trade-off

#  Build a list of values of exploratory rate. Decrease with times
#  Input : number of episodes (int), starting exploratory rate (float), exploratory rate decay (float),
#  minimum exploratory rate (float)
#  Output : list of exploratory rate (np.array)
def expRate_schedule(nE, exp_rate_start=1.0, exp_rate_decay=.9999, exp_rate_min=1e-4):
    x = np.arange(nE) + 1
    y = np.full(nE, exp_rate_start)
    y = np.maximum((exp_rate_decay ** x) * exp_rate_decay, exp_rate_min)
    return y
