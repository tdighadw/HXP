# DQN class inspired by : https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter25/lib/model.py
# ExperienceBuffer class inspired by : https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/02_dqn_pong.py

import torch
import torch.nn as nn
import numpy as np
import collections

#  Experience tuple used for DQN's training
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class DQN(nn.Module):
    def __init__(self, view_shape, feats_shape, n_actions):
        super(DQN, self).__init__()
        #  Input shape
        input_shape = (1, view_shape[0], view_shape[1])
        #  CNN part
        self.view_conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=2, padding=1),
            nn.ReLU(),
        )
        view_out_size = self._get_conv_out(input_shape)
        #  FC part
        self.fc = nn.Sequential(
            nn.Linear(view_out_size + feats_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    #  Pass through the CNN
    #  Input: shape of view datas (int Tuple)
    #  Output: output layer (np array)
    def _get_conv_out(self, shape):
        o = self.view_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    #  Pass through the whole DQN
    #  Input: input features (float list list)
    #  Output: output layer (float list)
    def forward(self, x):
        view_batch, feats_batch = x
        batch_size = view_batch.size()[0]
        #  CNN pass
        conv_out = self.view_conv(view_batch).view(batch_size, -1)
        #  Concatenate two tensors in a given dimension
        if batch_size == 1:
            fc_input = torch.cat((conv_out, feats_batch.unsqueeze(0)), dim=1)
        else:
            fc_input = torch.cat((conv_out, feats_batch), dim=1)
        #  FC pass
        return self.fc(fc_input)

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


        return states, np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               next_states

""" Method(s) used for calculating the MSE loss of a batch """

#  Calculate the loss
#  Input : batch (Experience list), NN (DQN), target NN (DQN), gamma (float), device (string),
#  use of Double-Q-learning (boolean)
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
    views = [state[0] for state in states]
    #  Normalization
    views = [normalize(view, 3, "view") for view in views]
    feats = [normalize(state[1], 9, "feats") for state in states]

    #  Convert into tensors
    tens_views = torch.tensor(np.array(views), dtype=torch.float32, device=device)
    tens_feats = torch.tensor(np.array(feats), dtype=torch.float32, device=device)

    return tens_views.unsqueeze(1), tens_feats

#  Normalize datas between 0 and 1
#  Input: datas (int list list), max possible value in datas (int) and type of data (String)
#  Output: normalized datas (float list list)
def normalize(datas, max_value, type):
    norm_datas = []
    # Neighbourhood
    if type == "view":
        for i in range(len(datas)):
            temp_list = []
            for j in range(len(datas[0])):
                temp_list.append(datas[i][j] / max_value)
            norm_datas.append(temp_list)

    # Position
    else:
        for i in range(len(datas)):
            if i == 2:
                norm_datas.append(datas[i] / 4)
            else:
                norm_datas.append(datas[i] / max_value)

    return norm_datas
