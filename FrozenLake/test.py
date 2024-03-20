import csv
import os
import queue
import sys
import time

import numpy as np
import argparse
from env import MyFrozenLake
from agent import Agent

# Get access to the HXP file
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from HXP import HXP
from HXP_tools import win, avoid_holes, holes, specific_part, specific_state, redefined_predicate
from HXP_tools import transition, terminal, render, preprocess, get_actions, constraint, sample
from HXP_tools import valid_history


if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-map', '--map_name', default="4x4", help="Map's dimension (nxn)", type=str, required=False)
    parser.add_argument('-policy', '--policy_name', default="4x4", help="Common part of policies name", type=str, required=False)
    parser.add_argument('-ep', '--nb_episodes', default=1, help="Number of episodes for a classic test of agent's policy", type=int, required=False)
    parser.add_argument('-k', '--length_k', default=5, help="Length of SXps or History", type=int, required=False)
    parser.add_argument('-csv', '--csv_filename', default="scores.csv", help="csv file to store utility from an HXp", type=str, required=False)

    parser.add_argument('-HXp', '--HXp', dest="COMPUTE_HXP", action="store_true", help="Compute HXp", required=False)
    parser.add_argument('-no_HXp', '--no_HXp', action="store_false", dest="COMPUTE_HXP", help="Do not compute HXp", required=False)
    parser.set_defaults(COMPUTE_HXP=True)

    parser.add_argument('-find_histories', '--find_histories', dest="find_histories", action="store_true", help="Find n histories", required=False)
    parser.add_argument('-no_find_histories', '--no_find_histories', action="store_false", dest="find_histories", help="Don't look for n histories", required=False)
    parser.set_defaults(find_histories=False)

    parser.add_argument('-equiprobable', '--equiprobable', dest="equiprobable", action="store_true", help="Equiprobable transitions", required=False)
    parser.add_argument('-no_equiprobable', '--no_equiprobable', action="store_false", dest="equiprobable", help="Equiprobable transitions", required=False)
    parser.set_defaults(equiprobable=False)

    parser.add_argument('-features', '--many_features', dest="many_features", action="store_true", help="Several features to define agent's state", required=False)
    parser.add_argument('-no_features', '--no_many_features', action="store_false", dest="many_features", help="Only one feature to define the agent's state", required=False)
    parser.set_defaults(many_features=False)

    parser.add_argument('-imp_type', '--importance_type', default="action", help="To compute HXp without user queries, choice of the type of importance to search", type=str, required=False)
    parser.add_argument('-pre', '--predicate', default="goal", help="Predicate to verify in the history", type=str, required=False)
    parser.add_argument('-n', '--n', default=1, help="Most important action to highlight", type=int, required=False)
    parser.add_argument('-spec_his', '--specific_history', default='', help="Express the specific history", type=str, required=False)
    parser.add_argument('-strat', '--strategy', default="exh", help="Exploration strategy for generating HXp", type=str, required=False)
    parser.add_argument('-strats', '--strategies', default="", help="Exploration strategies for similarity measures", type=str, required=False)
    parser.add_argument('-backward', '--backward', default='', help="Specify (max) horizon and delta for backward HXP",
                        type=str, required=False)
    parser.add_argument('-coefs', '--coefficents', default='', help="coefficients for B-HXP computation with two studied predicates",
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
    MAP_NAME = args.map_name
    POLICY_NAME = args.policy_name
    EQUIPROBABLE = args.equiprobable
    K = args.length_k

    NUMBER_EPISODES = args.nb_episodes
    CSV_FILENAME = args.csv_filename
    STRATEGY = args.strategy
    COMPUTE_HXP = args.COMPUTE_HXP
    PREDICATE = args.predicate
    N = args.n
    IMPORTANCE_TYPE = args.importance_type
    FIND_HISTORIES = args.find_histories
    STRATEGIES = args.strategies
    FEATURES = args.many_features

    backward = args.backward
    coefs = args.coefficents
    history_file = args.specific_history

    NB_SAMPLE = args.paxp_samples
    SELECT = args.backward_select

    FIXED_HORIZON = args.fixed_horizon

    #  Fill the specific history list (convert string into int list)
    SPECIFIC_HISTORY = []
    if COMPUTE_HXP and history_file != "":
        history_file = 'Histories' + os.sep + history_file
        file = open(history_file, 'r')
        lines = file.readlines()
        cpt = 0

        for idx, line in enumerate(lines):
            # state with several features
            if line[0] == '(':
                tmp_row = tuple(int(t) for t in line[1:-2].split(', '))
                SPECIFIC_HISTORY.append(tmp_row)
            # classic state / action
            else:
                SPECIFIC_HISTORY.append(int(line))
        print("Specific history : {}".format(SPECIFIC_HISTORY))

    # Path to obtain the Q table
    agent_Q_dirpath = "Q-tables" + os.sep + "Agent"

    # Path to store actions utility in case of HXp
    if COMPUTE_HXP:
        utility_dirpath = 'Utility' + os.sep + MAP_NAME
        if not os.path.exists(utility_dirpath):
            os.mkdir(utility_dirpath)
        utility_csv = utility_dirpath + os.sep + CSV_FILENAME
    else:
        utility_csv = 'scores.csv'

    # Path to store histories
    if COMPUTE_HXP and FIND_HISTORIES:
        tmp_str = '-histories-' if not FEATURES else '-histories-feats-'
        hist_dirpath = 'Histories' + os.sep + str(NUMBER_EPISODES)+tmp_str+MAP_NAME

        if not os.path.exists(hist_dirpath):
            os.mkdir(hist_dirpath)
        hist_csv = hist_dirpath + os.sep + CSV_FILENAME
    else:
        hist_csv = 'trash.csv'

    #  Envs initialisation
    if EQUIPROBABLE:
        env = MyFrozenLake(map_name=MAP_NAME, many_features=FEATURES)
    else:
        env = MyFrozenLake(map_name=MAP_NAME, slip_probas=[0.2, 0.6, 0.2], many_features=FEATURES)

    #  Agent initialization
    agent = Agent(POLICY_NAME, env)

    #  Load Q table
    agent.load(agent_Q_dirpath)

    # Initialize HXP class
    if COMPUTE_HXP:
        predicates = {'win': win,
                      'avoid_holes': avoid_holes,
                      'holes': holes,
                      'specific_state': specific_state,
                      'specific_part': specific_part,
                      'redefined_predicate': redefined_predicate}
        functions = [transition, terminal, constraint, sample, preprocess, get_actions, render]
        add_info = {'select': SELECT, 'nb_sample': NB_SAMPLE, 'fixed_horizon': FIXED_HORIZON}
        hxp = HXP('FL', agent, env, predicates, functions, add_info)

    # Convert horizon and delta type for B-HXP
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


    # Display agent's policy
    '''
    current = None
    q_vals = set()
    for key, values in agent.Q.items():

        if current != key[0]:
            if current is None:
                current = key[0]
            else:
                print('From position {}, actions {}'.format(current, q_vals))
                print()
                q_vals = set()
                current = key[0]
        q_vals.add(np.argmax(values))

        print('From state {} -- Action {}'.format(key, np.argmax(values)))
        print("-------------------------------------")

    print('From position {}, actions {}'.format(current, q_vals))
    '''

    # Compute (B-)HXP for a specific history
    if SPECIFIC_HISTORY and COMPUTE_HXP:
        specific_history = queue.Queue(maxsize=K * 2 + 1)

        for sa in SPECIFIC_HISTORY:  # specific_list
            specific_history.put(sa)

        if not STRATEGIES:
            hxp.explain('no_user', specific_history, PREDICATE, [STRATEGY], N, IMPORTANCE_TYPE, utility_csv, BACKWARD, COEFS)
        else:
            hxp.explain('compare', specific_history, PREDICATE, STRATEGIES, N, IMPORTANCE_TYPE, utility_csv, BACKWARD, COEFS)

    # Find histories
    elif FIND_HISTORIES and COMPUTE_HXP:
        nb_scenarios = NUMBER_EPISODES
        storage = []
        name, params = hxp.extract(PREDICATE)
        info = {'pred_params': params, 'env': env, 'agent': agent}

        # interaction loop
        while len(storage) != nb_scenarios:
            history = queue.Queue(maxsize=K * 2 + 1)
            obs = env.reset()
            done = False
            history.put(obs)  # initial state

            while not done:
                action = agent.predict(obs)
                if history.full():
                    history.get()
                    history.get()
                history.put(action)
                obs, reward, done, _ = env.step(action)
                history.put(obs)

                if valid_history(obs, name, info) and history.full():
                    data = [list(history.queue)]
                    storage.append(data)
                    if len(storage) == nb_scenarios:  # deal with specific_parts predicate (more than 1 history per episode)
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
        sum_reward = 0
        misses = 0
        steps_list = []
        nb_episode = NUMBER_EPISODES

        for episode in range(1, nb_episode + 1):
            obs = env.reset()
            done = False
            score = 0
            steps = 0
            if COMPUTE_HXP:
                history = queue.Queue(maxsize=K*2+1)

            while not done:

                steps += 1
                env.render()
                action = agent.predict(obs)

                #  Compute HXP
                if COMPUTE_HXP:
                    history.put(obs)
                    if steps != 1 and steps >= N + 1:
                        hxp.explain('user', history, approaches=[STRATEGY], n=N, imp_type=IMPORTANCE_TYPE, csv_file=utility_csv, backward=BACKWARD, coefs=COEFS)

                    # Update history
                    if history.full():
                        # Update history
                        history.get()
                        history.get()
                    history.put(action)

                obs, reward, done, info = env.step(action)
                score += reward

                # Store infos
                if done and reward == 1:
                    steps_list.append(steps)
                elif done and reward == 0:
                    misses += 1

            sum_reward += score
            print('Episode:{} Score: {}'.format(episode, score))

        if nb_episode > 1:
            print('Score: {}'.format(sum_reward/nb_episode))
            print('----------------------------------------------')
            print('Average of {:.0f} steps to reach the goal position'.format(np.mean(steps_list)))
            print('Fall {:.2f} % of the times'.format((misses / nb_episode) * 100))
            print('----------------------------------------------')
