# DQN class inspired by : https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter25/lib/model.py
# ExperienceBuffer class inspired by : https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/02_dqn_pong.py

# Network architecture : https://codebox.net/pages/connect4

import torch
import torch.nn as nn
import numpy as np
import collections

#  Experience tuple used for DQN's training
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class DQN(nn.Module):
    def __init__(self, view_shape, n_actions, arch=1):
        super(DQN, self).__init__()
        #  Input shape
        input_shape = (1, view_shape[0], view_shape[1])
        if not arch:
            #  CNN part
            self.view_conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 128, kernel_size=4),
                nn.ReLU(),
            )
        else:
            #  CNN part
            self.view_conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 128, kernel_size=4),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=2),
                nn.ReLU(),
            )

        view_out_size = self._get_conv_out(input_shape)
        #  FC part
        self.fc = nn.Sequential(
            nn.Linear(view_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
            )

    #  Pass through the CNN
    #  Input: shape of view datas (int tuple)
    #  Output: output layer (np array)
    def _get_conv_out(self, shape):
        o = self.view_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    #  Pass through the whole DQN
    #  Input: input features (float list list)
    #  Output: output layer (float list)
    def forward(self, x):
        batch_size = x.size()[0]
        #  CNN pass
        conv_out = self.view_conv(x).view(batch_size, -1)
        #  FC pass
        return self.fc(conv_out)

    #  Save DQN model in a bat file
    #  Input: filepath (String)
    #  Output: DQN model saved
    def save(self, path):
        return torch.save(self.state_dict(), path)

    #  Load DQN model
    #  Input: filepath (String)
    #  Output: DQN model (DQN)
    def load(self, path):
        return self.load_state_dict(torch.load(path))


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    #  Sample uniformly experiences from the buffer
    #  Input : size of the batch (int)
    #  Output : states, rewards, dones, new states (np arrays)
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])

        return states, np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), \
            next_states

""" Method(s) used for calculating the MSE loss of a batch """

#  Calculate the loss
#  Input : batch (Experience list), NN (DQN), target NN (DQN), gamma (float), device (string), use of Double-Q-learning
#  (bool)
#  Output : MSE loss
def calc_loss(batch, net, tgt_net, gamma, device="cpu", double=False):
    states, actions, rewards, dones, next_states = batch
    #  Convert into Tensors
    states_v = preprocess(states, device)
    next_states_v = preprocess(next_states, device)
    actions_v = torch.tensor(actions, dtype=torch.int64, device=device)
    rewards_v = torch.tensor(rewards, device=device)
    done_mask = torch.tensor(dones, dtype=torch.bool, device=device)

    #  Get state-action values for all experiences in the batch
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        if double:
            next_states_acts = net(next_states_v).max(1)[1]
            next_states_acts = next_states_acts.unsqueeze(-1)
            next_state_values = tgt_net(next_states_v).gather(1, next_states_acts).squeeze(-1)
        else:
            next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

        #  Compute target
        expected_state_action_values = next_state_values * gamma + rewards_v

    #  Compute MSE
    return nn.MSELoss()(state_action_values, expected_state_action_values), None

#  Normalize a batch of datas and convert them into Tensors
#  Input: set of states (int list list list) and type of device (String)
#  Output: two Tensors, one for each part of states (Tensor)
def preprocess(states, device="cpu"):
    #  Convert into tensor
    tens_views = torch.tensor(np.array(states), dtype=torch.float32, device=device)

    return tens_views.unsqueeze(1)
