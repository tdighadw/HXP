import csv
import queue
import torch
import os
import argparse
from copy import deepcopy
import sys

from DQN import DQN
from agent import Agent
from env import Connect4

# Get access to the HXP file
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from HXP import HXP
from HXP_tools import win, lose, control_midcolumn, align3, counteralign3, redefined_predicate
from HXP_tools import transition, terminal, render, preprocess, get_actions, constraint, sample
from HXP_tools import valid_history

###### AGENT's POLICY #################
'''
best model: bestPlayerP1_98_P2_96.dat
other model: badPlayer_82.dat
best without saving multiple NNs to change the opponent during the training:
   - tl1000000e800000s5000ddqnTrue-last.dat
   - tl1000000e800000s5000ddqnTruewindow10lmr5pc100000swap25000save20000-980000_steps.dat
'''
######################################

if __name__ == "__main__":

    #  Parser ----------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model_dir', default="Models"+os.sep+"bestPlayerP1_98_P2_96.dat", help="Agent's model", type=str, required=False)
    parser.add_argument('-r', '--render', action="store_true", dest="render", help="Environment rendering at each step", required=False)
    parser.add_argument('-no_r', '--no_render', action="store_false", dest="render", help="No environment rendering at each step", required=False)
    parser.set_defaults(render=False)
    parser.add_argument('-ep', '--nb_episodes', default=1,
                        help="Number of episodes for a classic test of agent's policy", type=int, required=False)
    parser.add_argument('-csv', '--csv_filename', default="scores.csv", help="csv file to store scores in case of starting from a specific state", type=str, required=False)
    parser.add_argument('-k', '--length_k', default=5, help="Length of SXps", type=int, required=False)
    parser.add_argument('-HXp', '--HXp', dest="COMPUTE_HXP", action="store_true", help="Compute HXp", required=False)
    parser.add_argument('-no_HXp', '--no_HXp', action="store_false", dest="COMPUTE_HXP", help="Do not compute HXp", required=False)
    parser.set_defaults(COMPUTE_HXP=True)
    parser.add_argument('-pre', '--predicate', default="win", help="Predicate to verify in the history", type=str, required=False)
    parser.add_argument('-spec_his', '--specific_history', default='', help="Express the specific history", type=str, required=False)
    parser.add_argument('-strat', '--strategy', default="exh", help="Type of strategy for (approximate) HXp",
                        type=str, required=False)

    parser.add_argument('-find_histories', '--find_histories', dest="find_histories", action="store_true", help="Find n histories", required=False)
    parser.add_argument('-no_find_histories', '--no_find_histories', action="store_false", dest="find_histories", help="Don't look for n histories", required=False)
    parser.set_defaults(find_histories=False)
    parser.add_argument('-rand', '--random', dest="random", action="store_true", help="Player 2 performs random choices", required=False)
    parser.add_argument('-no_rand', '--no_random', action="store_false", dest="random", help="Player 2 doesn't perform random choices", required=False)
    parser.set_defaults(random=True)
    parser.add_argument('-strats', '--strategies', default="", help="Exploration strategies for similarity measures", type=str, required=False)
    parser.add_argument('-n', '--n', default=1, help="Most important action to highlight", type=int, required=False)
    parser.add_argument('-imp_type', '--importance_type', default="action",
                        help="To compute HXp without user queries, choice of the type of importance to search",
                        type=str, required=False)

    parser.add_argument('-backward', '--backward', default='', help="Specify (max) horizon and delta for backward HXP",
                        type=str, required=False)
    parser.add_argument('-coefs', '--coefficents', default='',
                        help="coefficients for B-HXP computation with two studied predicates",
                        type=str, required=False)

    parser.add_argument('-samples', '--paxp_samples', default=-1,
                        help="Maximal number of state samples for the evaluation of a feature during the PAXp computation", type=int, required=False)
    parser.add_argument('-select', '--backward_select', default='imp', help="Method to select the important state of a sub-sequence (backward HXP)",
                        type=str, required=False)

    parser.add_argument('-fixed_horizon', '--fixed_horizon', dest="fixed_horizon", action="store_true",
                        help="Utility: probability to respect the predicate at horizon k", required=False)
    parser.add_argument('-unfixed_horizon', '--unfixed_horizon', action="store_false", dest="fixed_horizon",
                        help="Utility: probability to respect the predicate at maximal horizon k", required=False)
    parser.set_defaults(fixed_horizon=True)

    args = parser.parse_args()

    # Get arguments
    PATHFILE_MODEL = args.model_dir
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    K = args.length_k
    NUMBER_EPISODES = args.nb_episodes
    COMPUTE_HXP = args.COMPUTE_HXP
    STRATEGY = args.strategy
    PREDICATE = args.predicate
    CSV_FILENAME = args.csv_filename
    RENDER = args.render
    history_file = args.specific_history
    SPECIFIC_HISTORY = []
    FIND_HISTORIES = args.find_histories
    N = args.n
    STRATEGIES = args.strategies
    RANDOM = args.random
    IMPORTANCE_TYPE = args.importance_type

    backward = args.backward
    coefs = args.coefficents
    NB_SAMPLE = args.paxp_samples
    SELECT = args.backward_select

    FIXED_HORIZON = args.fixed_horizon

    #  Fill pre-defined history
    if COMPUTE_HXP and history_file != "":
        file = open('Histories' + os.sep + history_file, 'r')
        lines = file.readlines()
        state = []
        cpt = 0
        for idx, line in enumerate(lines):
            if idx % 2:
                SPECIFIC_HISTORY.append(int(line[:-1]))
            else:
                tmp_row = [[int(t) for t in sublist.split(', ')] for sublist in line[2:-3].split('], [')]
                SPECIFIC_HISTORY.append(tmp_row)

    # Path to store actions utility in case of HXp
    if COMPUTE_HXP:
        utility_dirpath = 'Utility'
        if not os.path.exists(utility_dirpath):
            os.mkdir(utility_dirpath)
        utility_csv = utility_dirpath + os.sep + CSV_FILENAME
    else:
        utility_csv = 'scores.csv'

    # Path to store similarity measure in case of HXp
    if COMPUTE_HXP and FIND_HISTORIES:
        hist_dirpath = 'Histories' + os.sep + str(NUMBER_EPISODES)+'-histories'
        if not os.path.exists(hist_dirpath):
            os.mkdir(hist_dirpath)
        hist_csv = hist_dirpath + os.sep + CSV_FILENAME
    else:
        hist_csv = 'trash.csv'

    #  Initialization --------------------------------------------------------------------------------------------------

    #  Env
    env = Connect4()
    #  Agents
    player_1 = Agent('Yellow', env)
    player_2 = Agent('Red', env, random=RANDOM)
    agents = [player_1, player_2]

    #  Load net(s)
    net = DQN((env.rows, env.cols), env.action_space.n).to(DEVICE)
    net.load_state_dict(torch.load(PATHFILE_MODEL, map_location=DEVICE))

    # Initialize HXP class
    if COMPUTE_HXP:
        predicates = {'win': win,
                      'lose': lose,
                      'control_midcolumn':  control_midcolumn,
                      'align3': align3,
                      'counteralign3': counteralign3,
                      'redefined_predicate': redefined_predicate}
        functions = [transition, terminal, constraint, sample, preprocess, get_actions, render]
        initial_s = SPECIFIC_HISTORY[0] if SPECIFIC_HISTORY else None
        add_info = {'opponent': player_2, 'net': net, 'initial_s': initial_s, 'select': SELECT, 'nb_sample': NB_SAMPLE, 'fixed_horizon': FIXED_HORIZON}
        hxp = HXP('C4', player_1, env, predicates, functions, add_info)

    # Convert horizon and delta type for backward HXP
    if backward:
        str_backward = backward.split(',')
        BACKWARD = [int(str_backward[0]), float(str_backward[1])]
        print('BACKWARD: {}'.format(BACKWARD))
    else:
        BACKWARD = []

    # Convert B-HXP predicate coefficients
    if coefs:
        str_coefs = coefs.split(',')
        COEFS = [float(str_coefs[0]), float(str_coefs[1])]
    else:
        COEFS = []

    #  Test ------------------------------------------------------------------------------------------------------------

    # Compute (B-)HXP for a specific history
    if SPECIFIC_HISTORY and COMPUTE_HXP:
        specific_history = queue.Queue(maxsize=K * 2 + 1)
        for sa in SPECIFIC_HISTORY:  # specific_list
            specific_history.put(sa)
        list_hist = list(specific_history.queue)
        env_copy = deepcopy(env)
        for idx, elm in enumerate(list_hist):
            print(elm)
            if idx % 2 != 1:
                env_copy.board = elm
                env_copy.render()
        if not STRATEGIES:
            hxp.explain('no_user', specific_history, PREDICATE, [STRATEGY], N, IMPORTANCE_TYPE, utility_csv, BACKWARD, COEFS)
        else:
            hxp.explain('compare', specific_history, PREDICATE, STRATEGIES, N, IMPORTANCE_TYPE, utility_csv, BACKWARD, COEFS)

    # Find histories
    elif FIND_HISTORIES and COMPUTE_HXP:
        nb_scenarios = NUMBER_EPISODES
        storage = []
        info = {'env': env, 'agent': player_1, 'initial_s': None}

        # interaction loop
        while len(storage) != nb_scenarios:
            history = queue.Queue(maxsize=K * 2 + 1)
            #  Reset env
            state = env.reset()
            done = False
            history.put(deepcopy(state))  # initial state
            while not done:
                #  Choose action
                action = player_1.predict(state, net, device=DEVICE)
                #  History update
                if history.full():
                    history.get()
                    history.get()
                history.put(deepcopy(action))
                #  Step
                reward, done, state, _, _ = env.step(agents, action, epsilon=0.3, net=net, device=DEVICE)
                #  History update
                history.put(deepcopy(state))
                info['initial_s'] = list(history.queue)[0]
                if valid_history(state, PREDICATE, info) and history.full():
                    data = [list(history.queue)]
                    storage.append(data)
                    if len(storage) == nb_scenarios:
                        break

        # Store infos into CSV
        with open(hist_csv, 'a') as f:
            writer = csv.writer(f)
            # First Line
            line = ['History']
            writer.writerow(line)
            # Data
            for data in storage:
                writer.writerow(data)

    # Classic testing loop
    else:
        rewards = []
        s_a_list = []
        episodes = NUMBER_EPISODES
        lose = False
        cpt = 0

        while not lose:
            done = False
            env.reset()
            env.render()
            if COMPUTE_HXP:
                history = queue.Queue(maxsize=K * 2 + 1)
            cpt = 0

            while not done:
                #  Choose action
                state = env.board
                s_a_list.append(deepcopy(state))
                action = player_1.predict(state, net, device=DEVICE)
                s_a_list.append(action)
                #  HXP
                if COMPUTE_HXP:
                    history.put(s_a_list[-2])
                    # Compute HXp
                    if cpt and cpt >= K:
                        hxp.add_info['initial_s'] = list(history.queue)[0]
                        hxp.explain('user', history, approaches=[STRATEGY], n=N, imp_type=IMPORTANCE_TYPE, csv_file=utility_csv, backward=BACKWARD, coefs=COEFS)
                    # History update
                    if history.full():
                        history.get()
                        history.get()
                    history.put(s_a_list[-1])
                #  Step and update observation
                reward, done, new_state, _, _ = env.step(agents, action, epsilon=0.1, net=net, device=DEVICE)
                #  Render
                env.render()
                if done:  # and history.full()
                    s_a_list = []
                    rewards.append((reward+1)/2)
                    if env.win(-1) and not env.win(1):
                        lose = True
                    #print('length: ', len(list(history.queue)))
                cpt += 1

            # Compute last HXp
            if COMPUTE_HXP:
                history.put(new_state)
                hxp.add_info['initial_s'] = list(history.queue)[0]
                hxp.explain('user', history, approaches=[STRATEGY], n=N, imp_type=IMPORTANCE_TYPE, csv_file=utility_csv, backward=BACKWARD, coefs=COEFS)

        if episodes > 1:
            print(sum(rewards))
            print((sum(rewards)/len(rewards))*100)
            print('Win rate: {}% over {} episodes vs Random agent'.format(((sum(rewards)/len(rewards))*100), episodes))
