import time
from copy import deepcopy

import numpy as np
import argparse
import collections
import os
from collections import deque
from tqdm import tqdm
from DQN import DQN, ExperienceBuffer, calc_loss
from agent import Agent
from env import Connect4
import torch.optim as optim
import torch
from distutils.util import strtobool

#  Experience tuple used for DQN's training
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


#  Compute a training of both agents (First and Second player of Connect4)
#
#  Input: filepaths of logs, models and logfile (str), environment (Connect4), agents (Agent list),
#  NN and target NN (DQN),  experience buffers (Experience list list), learning rate (float), epsilon limit, and it's
#  decay rate (float), type of device (str), timestep where the NN training begins (int), frequency of target NN
#  update (int), batch size (int), discount factor (loss calculation) (float), time limit of training (int), use of
#  double-Q learning extension (boolean), list of saved models (str list list), number of steps before changing the
#  learning agent (int), number of steps before changing the DQN used for training the learning agent (int) and
#  the ratio of the latest trained model (float)
#
#  Output: averages of losses, rewards (float list list), ended time episode (int list list) and the file name of
#  the best NN produced (str)
def agentTrain(log_dir, model_dir, filenames, env, agents, nets, tgt_nets, buffers, learning_rate, epsilon_final, epsilon_decay, device, replay_start, sync_target, batch_size, gamma, time_limit, ddqn, models, player_change, swap_steps, latest_model_ratio):
    # Initialization
    timestep = 0  # Global timestep
    timestep1 = 0
    timestep2 = 0
    epstep = 0  # Episode step
    nb_episode = 0  # Number of episodes
    start_time = time.time()
    file_p1 = open(log_dir + os.sep + filenames[0] + ".txt", "a")
    file_p2 = open(log_dir + os.sep + filenames[1] + ".txt", "a")

    #  Player1, Player2
    player_1, player_2 = agents
    learning_agent = 'Yellow'
    #  Player2 list to store info for creating Experiences
    player_2_experience = []
    #  Net(s)
    net1 = nets[0]
    net2 = nets[1]
    #  Save initial models
    model1 = model_dir + os.sep + filenames[0] + "-init.dat"
    model2 = model_dir + os.sep + filenames[1] + "-init.dat"
    torch.save(net1.state_dict(), model1)
    torch.save(net2.state_dict(), model2)
    models[0].append(model1)
    models[1].append(model2)
    #  Buffers
    buffer1, buffer2 = buffers

    losses = [[], []]  # Store loss values
    total_rewards = [[], []]  # Store all rewards obtained at the end of an episode
    ended_time_horizon = [[], []]  # An episode end with time horizon limit (bool)
    mean_losses = [[[], []], [[], []]]   # Store averages of 100 losses and timestep where they were calculated
    mean_rewards = [[[], []], [[], []]]  # Store averages of 100 cumulative reward and timestep where they were calculated
    mean_time_horizon = [[[], []], [[], []]]  # Store averages of 100 ended time episode and timestep where they were calculated

    #  Initialize optimizer and epsilon rate
    optimizer_p1 = optim.Adam(nets[0].parameters(), lr=learning_rate)
    optimizer_p2 = optim.Adam(nets[1].parameters(), lr=learning_rate)
    epsilon_rate = epsilon_final ** (1 / (epsilon_decay // 2))

    for i in tqdm(range(time_limit)):
        epstep += 1

        # Change learning player
        if not i % player_change and i:
            if learning_agent == 'Yellow':
                learning_agent = 'Red'
                net2 = nets[1]
            else:
                learning_agent = 'Yellow'
                net1 = nets[0]

        # Modify frozen policy
        if not i % swap_steps and i:
            if learning_agent == 'Yellow':
                frozen_net = select_net(models[1], latest_model_ratio, env, device)
                net2 = frozen_net
            else:
                frozen_net = select_net(models[0], latest_model_ratio, env, device)
                net1 = frozen_net

        #  Classic RL loop----------------------------------------------------------------------------------------------

        #  Trade-off exploration/exploitation
        tmp_timestep = timestep1 if learning_agent == 'Yellow' else timestep2
        epsilon = max(epsilon_final, epsilon_rate ** (tmp_timestep+1))

        #  Choose action
        state = deepcopy(env.board)
        action = player_1.predict(state, net1, epsilon, device=device)

        #  Step and update observation
        reward, done, new_state, player_2_experience, player_2_tmp_experience = env.step(agents, action,
                                                    player_2_experience, net2, epsilon, learning_agent, device=device)

        #  Save experiences depending on the learning agent-------------------------------------------------------------

        #  Player 1 Experiences
        if learning_agent == 'Yellow':
            timestep1 += 1
            buffer1.append(Experience(state, action, reward, done, new_state))

        #  Player 2 Experiences
        else:
            timestep2 += 1
            if len(player_2_experience) == 5 and not player_2.random:
                inverse_state = env.inverse_board(player_2_experience[0])
                inverse_new_state = env.inverse_board(player_2_experience[-1])
                reward_p2 = player_2_experience[2]
                buffer2.append(Experience(inverse_state, player_2_experience[1], player_2_experience[2], player_2_experience[3], inverse_new_state))
            player_2_experience = player_2_tmp_experience

        #  Store the last cumulative reward of an episode and reset env ------------------------------------------------
        if done:
            if learning_agent == 'Yellow':
                normed_reward = (reward - env.rewards['lose']) / (env.rewards['win'] - env.rewards['lose'])
                file_p1, total_rewards, ended_time_horizon, epstep, nb_episode, m_reward, m_endTimeHorizon = \
                    save_logs(normed_reward, env, file_p1, total_rewards, ended_time_horizon, epstep, nb_episode, timestep1, learning_agent)

                #  Store infos
                if m_reward is not None:
                    mean_rewards[0][0].append(m_reward)
                    mean_rewards[0][1].append(timestep1)
                    mean_time_horizon[0][0].append(m_endTimeHorizon)
                    mean_time_horizon[0][1].append(timestep)
            else:
                if len(buffer2) > 0:
                    normed_reward = (reward_p2 - env.rewards['lose']) / (env.rewards['win'] - env.rewards['lose'])
                    file_p2, total_rewards, ended_time_horizon, epstep, nb_episode, m_reward, m_endTimeHorizon = \
                        save_logs(normed_reward, env, file_p2, total_rewards, ended_time_horizon, epstep, nb_episode, timestep2, learning_agent)

                    #  Store infos
                    if m_reward is not None:
                        mean_rewards[1][0].append(m_reward)
                        mean_rewards[1][1].append(timestep2)
                        mean_time_horizon[1][0].append(m_endTimeHorizon)
                        mean_time_horizon[1][1].append(timestep)

        #  Save current NN
        if not i % save_steps and i:
            if learning_agent == 'Yellow':
                str_scores = test_model(10000, nets[0], deepcopy(env), learning_agent)
                model = model_dir + os.sep + filenames[0] + "-"+str(tmp_timestep)+"_steps" + str_scores + ".dat"
                torch.save(nets[0].state_dict(), model)
                models[0].append(model)

            elif i != player_change:
                str_scores = test_model(10000, nets[1], deepcopy(env), learning_agent)
                model = model_dir + os.sep + filenames[1] + "-"+str(tmp_timestep)+"_steps" + str_scores + ".dat"
                torch.save(nets[1].state_dict(), model)
                models[1].append(model)

        #  Skip updates if there is not enough experiences
        if learning_agent == 'Yellow' and len(buffer1) < replay_start:
            continue
        if learning_agent == 'Red' and len(buffer2) < replay_start:
            continue

        #  Update target NN(s) weights ---------------------------------------------------------------------------------
        if tmp_timestep % sync_target == 0:
            if learning_agent == 'Yellow':
                tgt_nets[0].load_state_dict(nets[0].state_dict())
            else:
                tgt_nets[1].load_state_dict(nets[1].state_dict())

        #  Update NN(s) ------------------------------------------------------------------------------------------------
        if learning_agent == 'Yellow':
            optimizer_p1.zero_grad()
            batch = buffer1.sample(batch_size)
            loss, _ = calc_loss(batch, nets[0], tgt_nets[0], gamma, device=device, double=ddqn)
            losses[0].append(loss.item())

            #  Store infos
            if len(losses[0]) >= 100:
                mean_losses[0][0].append(np.mean(losses[0][-100:]))
                mean_losses[0][1].append(tmp_timestep)
                losses[0] = []
            loss.backward()
            optimizer_p1.step()

        else:
            optimizer_p2.zero_grad()
            batch = buffer2.sample(batch_size)
            loss, _ = calc_loss(batch, nets[1], tgt_nets[1], gamma, device=device, double=ddqn)
            losses[1].append(loss.item())

            #  Store infos
            if len(losses[1]) >= 100:
                mean_losses[1][0].append(np.mean(losses[1][-100:]))
                mean_losses[1][1].append(tmp_timestep)
                losses[1] = []
            loss.backward()
            optimizer_p2.step()

    #  Save last models and display time information
    torch.save(nets[0].state_dict(), model_dir + os.sep + filenames[0] + "-final.dat")
    torch.save(nets[1].state_dict(), model_dir + os.sep + filenames[1] + "-final.dat")

    final_time_s = time.time() - start_time
    time_hour, time_minute, time_s = (final_time_s // 60) // 60, (final_time_s // 60) % 60, final_time_s % 60

    file_p1.write("Training process achieved in:\n {} hour(s)\n {} minute(s)\n {} second(s)".format(time_hour, time_minute, time_s))
    file_p2.write("Training process achieved in:\n {} hour(s)\n {} minute(s)\n {} second(s)".format(time_hour, time_minute, time_s))
    print("Training process achieved in:\n {} hour(s)\n {} minute(s)\n {} second(s)".format(time_hour, time_minute, time_s))

    #  Close log files
    file_p1.close()
    file_p2.close()

    return mean_losses, mean_rewards, mean_time_horizon

#  Save data into lists and a file, reset agents and store the current NN when a better cumulative reward is
#  reached. save() can stop the training process. This function is called at the end of an episode.
#
#  Input: reward (float), environment (Connect4), file to write data (str), list of cumulative reward and ended time
#  horizon (float list), ended timestep of the episode (int), number of finished episode (int), current timestep of the
#  training (int) and learning agent (str)
#
#  Output: file to write data (str), list of cumulative reward and ended time horizon (float list), ended timestep of
#  the episode (int), number of finished episode (int), the last average of the cumulative reward and
#  ended time horizon (float)
def save_logs(reward, env, file, total_rewards, ended_time_horizon, epstep, nb_episode, timestep, learning_agent):
    idx = {'Yellow': 0, 'Red': 1}[learning_agent]
    #  Store infos about completeness of an episode
    ended_time_horizon[idx].append(epstep)
    total_rewards[idx].append(reward)

    nb_episode += 1
    #  Reset env
    env.reset()
    epstep = 0

    if len(total_rewards[idx]) >= 100:
        #  Write mean reward in file
        m_reward = np.mean(total_rewards[idx][-100:])
        m_endTimeHorizon = np.mean(ended_time_horizon[idx][-100:])
        total_rewards[idx] = []
        ended_time_horizon[idx] = []

        file.write("Global timestep {} - Episodes achieved : {} - Mean of reward of the last 100 ep : {} - Mean end "
                   "with time horizon {} \n".format(timestep, nb_episode, m_reward, m_endTimeHorizon))

        return file, total_rewards, ended_time_horizon, epstep, nb_episode, m_reward, m_endTimeHorizon

    return file, total_rewards, ended_time_horizon, epstep, nb_episode, None, None

#  Write data into file
#  Input: filename (str), data (float list list)
#  Output: None
def storeData(filename, data):
    file = open(filename + ".txt", "a")
    file.write(str(data) + "\n")
    file.close()
    return

#  Store data in files. It's used at the end of a training to save the evolution of loss value, cumulative reward and
#  ended time horizon
#  Input: filenames (str list), data to store in filenames (float list list), data labels (str list)
#  and directory path (str)
#  Output: None
def saveInfos(filenames, datas, ylabels, dir):
    for i in range(len(filenames)):
        print("First {} : {} Last : {}".format(ylabels[i], datas[i][0][0], datas[i][0][-1]))
        storeData(dir + os.sep + filenames[i], datas[i])

    return

#  Select a NN during the training process. This is done to diversify the behaviors the agent learn to face
#  Input: queue of NN (str Queue), ratio of the latest model (float), environment (Connect4), type of device (str)
#  Output: the selected NN (DQN)
def select_net(queue, latest_model_ratio, env, device='cpu'):
    model_list = list(queue)

    if len(model_list) == 1:
        str_model = model_list[0]
    else:
        #  Extract a model
        tmp_ratio = [(1-latest_model_ratio)/(len(model_list)-1) for _ in range(len(model_list)-1)]
        tmp_ratio.extend([latest_model_ratio])
        str_model = np.random.choice(model_list, p=tmp_ratio)

    #  Instance it
    frozen_net = DQN((env.rows, env.cols), env.action_space.n).to(device)
    frozen_net.load_state_dict(torch.load(str_model, map_location=device))

    return frozen_net

#  Compute the win rate of a learning agent over n episodes
#  Input: number of episode (int), policy (DQN), environment (Connect4), learning agent (str)
#  Output: win rate (float)
def test_model(episodes, net, env, learning_agent):
    #  Player1 vs Random
    rewards = []
    if learning_agent == 'Yellow':
        player_1 = Agent('Yellow', env)
        player_2 = Agent('Red', env, random=True)
        agents = [player_1, player_2]

        for i in range(episodes):
            done = False
            env.reset()
            while not done:
                #  Choose action
                state = env.board
                action = player_1.predict(state, net, device=device)
                #  Step and update observation
                reward, done, new_state, _, _ = env.step(agents, action, net=net, device=device)
                if done:
                    rewards.append((reward + 1) / 2)

        p1_score = (sum(rewards) / len(rewards)) * 100
        print('P1 score: {}'.format(p1_score))
        return str(round(p1_score))

    #  Player2 vs Random
    else:
        rewards = []
        env.reset()
        player_1 = Agent('Yellow', env, random=True)
        player_2 = Agent('Red', env)
        agents = [player_1, player_2]

        for i in range(episodes):
            done = False
            env.reset()
            while not done:
                #  Choose action
                state = env.board
                action = player_1.predict(state, net, device=device)
                #  Step and update observation
                reward, done, new_state, _, _ = env.step(agents, action, net=net, device=device)
                if done:
                    rewards.append((reward + 1) / 2)

        p2_score = 100 - ((sum(rewards) / len(rewards)) * 100)
        print('P2 score: {}'.format(p2_score))
        return str(round(p2_score))

#  Create a symmetric Experience. It's a data augmentation tool
#  Input: environment (Connect4), experience (Experience)
#  Output: new experience (Experience)
def symmetric_experience(env, experience):
    state, action, reward, done, new_state = experience
    #  Reverse states
    sym_state = symmetric_state(state)
    sym_new_state = symmetric_state(new_state)
    #  Reverse action
    sym_action = env.action_space.n - (action+1)

    return Experience(sym_state, sym_action, reward, done, sym_new_state)

#  Provide a symmetric state (vertical symmetry)
#  Input: board (int list list)
#  Output: symmetric board (int list list)
def symmetric_state(state):
    symmetric_state = []
    for row in state:
        symmetric_state.append(list(reversed(row)))

    return symmetric_state

if __name__ == "__main__":

    #  Parser ----------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    #  Directory Paths
    parser.add_argument('-model', '--model_dir', default="", help="Agent's model directory", type=str, required=True)
    parser.add_argument('-log', '--log_dir', default="", help="Agent's log directory", type=str, required=True)

    #  Hyper-parameters
    parser.add_argument('-limit', '--timestep_limit', default=1000000, help="Limits for training", type=int, required=True)
    parser.add_argument('-eps_dec', '--epsilon_decay', default=800000, help="Number of steps where espilon decrease", type=int, required=False)
    parser.add_argument('-eps_f', '--epsilon_final', default=0.1, help="Epsilon' final value", type=float, required=False)
    parser.add_argument('-sd', '--seed', type=int, default=123, help='seed of the experiment')
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    #  Self Play params
    parser.add_argument('-window', '--window', default=10, help="Number of policy to store and use for the self learning", type=int, required=False)
    parser.add_argument('-latest_model_ratio', '--play_against_latest_model_ratio', default=0.5, help="Probability of an agent to play against the latest opponent policy", type=float, required=False)
    parser.add_argument('-save_steps', '--save_steps', default=20000, help="Every x steps, save the policy", type=int, required=False)
    parser.add_argument('-player_change', '--player_change', default=100000, help="Every x steps, the opponent player learns", type=int, required=False)
    parser.add_argument('-swap_steps', '--swap_steps', default=25000, help="Every x steps, the opponent player policy change", type=int, required=False)
    #  DQN params
    parser.add_argument('-batch', '--batch_size', default=32, help="Batch size", type=int, required=False)
    parser.add_argument('-lr', '--learning_rate', default=1e-6, help="Learning rate", type=float, required=False)
    parser.add_argument('-df', '--discount_factor', default=.99, help="Discount factor", type=float, required=False)
    parser.add_argument('-sync', '--sync_target', default=5000, help="Synchronize target net at each n steps", type=int, required=False)
    parser.add_argument('-replay', '--replay_size', default=30000, help="Size of replay memory", type=int, required=False)
    parser.add_argument('-replay_s', '--replay_starting_size', default=10000, help="From which number of experiences NN training process start", type=int, required=False)
    parser.add_argument('-ddqn', '--double_dqn', action="store_true", dest='ddqn', help="Use Double DQN upgrade", required=False)
    parser.add_argument('-no_ddqn', '--no_double_dqn', action="store_false", dest='ddqn', help="Use Double DQN upgrade", required=False)
    parser.set_defaults(ddqn=True)

    args = parser.parse_args()

    # Get arguments
    log_dir = args.log_dir
    model_dir = args.model_dir

    time_limit = args.timestep_limit

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    gamma = args.discount_factor
    epsilon_decay = args.epsilon_decay
    epsilon_final = args.epsilon_final

    sync_target = args.sync_target
    replay_size = args.replay_size
    replay_start = args.replay_starting_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ddqn = args.ddqn

    window = args.window
    latest_model_ratio = args.play_against_latest_model_ratio
    save_steps = args.save_steps
    player_change = args.player_change
    swap_steps = args.swap_steps

    filename_p1 = "p1_lr" + str(learning_rate) + "s" + str(sync_target) + "ddqn" + str(ddqn) + "window"\
               + str(window) + "bs" + str(batch_size) + "pc" + str(player_change) + "swap" + str(swap_steps)\
               + "save" + str(save_steps) + "eps_dec_" + str(epsilon_decay)

    filename_p2 = "p2_lr" + str(learning_rate) + "s" + str(sync_target) + "ddqn" + str(ddqn) + "window"\
               + str(window) + "bs" + str(batch_size) + "pc" + str(player_change) + "swap" + str(swap_steps)\
               + "save" + str(save_steps) + "eps_dec_" + str(epsilon_decay)
    filenames_p = [filename_p1, filename_p2]

    #  Create Dirs to store logs and models
    if not os.path.exists(log_dir):  # log dir
        os.mkdir(log_dir)
    if not os.path.exists(model_dir):  # model dir
        os.mkdir(model_dir)

    #  Initialization --------------------------------------------------------------------------------------------------

    #  Env
    env = Connect4()

    #  Agents
    player_1 = Agent('Yellow', env)
    player_2 = Agent('Red', env)
    agents = [player_1, player_2]

    #  Policy : NN and target NN
    net_p1 = DQN((env.rows, env.cols), env.action_space.n).to(device)
    tgt_net_p1 = DQN((env.rows, env.cols), env.action_space.n).to(device)

    net_p2 = DQN((env.rows, env.cols), env.action_space.n).to(device)
    tgt_net_p2 = DQN((env.rows, env.cols), env.action_space.n).to(device)

    nets = [net_p1, net_p2]
    tgt_nets = [tgt_net_p1, tgt_net_p2]

    #  Init models Queue
    models_p1 = deque(maxlen=window)
    models_p2 = deque(maxlen=window)
    models = [models_p1, models_p2]

    #  ExperienceBuffer
    buffer1 = ExperienceBuffer(replay_size)
    buffer2 = ExperienceBuffer(replay_size)
    buffers = [buffer1, buffer2]

    #  Save infos of Agent's training
    filenames_log1 = ['losses_' + filename_p1, 'rewards_' + filename_p1, 'time_horizons_' + filename_p1]
    filenames_log2 = ['losses_' + filename_p2, 'rewards_' + filename_p2, 'time_horizons_' + filename_p2]
    ylabels = ["Loss", "Mean reward", "Mean ended time horizon"]

    #  Train -----------------------------------------------------------------------------------------------------------
    losses, rewards, timeH = agentTrain(log_dir, model_dir, filenames_p, env, agents, nets, tgt_nets, buffers,
    learning_rate, epsilon_final, epsilon_decay, device, replay_start, sync_target, batch_size, gamma, time_limit, ddqn,
                                                        models, player_change, swap_steps, latest_model_ratio)
    datas_p1 = [losses[0], rewards[0], timeH[0]]
    datas_p2 = [losses[1], rewards[1], timeH[1]]
    saveInfos(filenames_log1, datas_p1, ylabels, model_dir)
    saveInfos(filenames_log2, datas_p2, ylabels, model_dir)
