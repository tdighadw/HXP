import csv
import queue
from copy import deepcopy

import torch
import os
import sys
import argparse
import numpy as np

from DQN import DQN
from agent import Agent
from env import DroneAreaCoverage

# Get access to the HXP file
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from HXP import HXP
from HXP_tools import perfect_cover, imperfect_cover, crash, no_drones, max_reward, region, redefined_predicate
from HXP_tools import transition, terminal, render, preprocess, get_actions, constraint, sample
from HXP_tools import valid_history
from HXP_tools import get_transition_exhaustive_lists

STOP = 4

if __name__ == "__main__":

    #  Parser ----------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    #  Path
    parser.add_argument('-model', '--model_dir', default="Models"+os.sep+"Agent"+os.sep+"tl1600000e750000s50000th22ddqnTrue-best_11.69.dat", help="Agent's model", type=str, required=False)

    parser.add_argument('-map', '--map_name', default="10x10", help="Map's name", type=str, required=False)
    parser.add_argument('-agents', '--number_agents', default=4, help="Number of agents in the map", type=int, required=False)
    parser.add_argument('-ep', '--nb_episodes', default=1,
                        help="Number of episodes for a classic test of agent's policy", type=int, required=False)
    parser.add_argument('-horizon', '--time_horizon', default=20, help="Time horizon of an episode", type=int, required=False)
    parser.add_argument('-rand', '--random_starting_position', action="store_true", dest='random_starting_position', help="At the beginning of an episode, each drone start at random positions", required=False)
    parser.add_argument('-no_rand', '--no_random_starting_position', action="store_false", dest='random_starting_position', help="At the beginning of an episode, each drone start at random positions", required=False)
    parser.set_defaults(random_starting_position=True)
    parser.add_argument('-move', '--step_move', default="stop", help="Type of transition with wind", type=str, required=False)
    parser.add_argument('-view', '--view_range', default=5, help="View range of a drone", type=int, required=False)
    parser.add_argument('-w', '--wind', action="store_false", dest='windless', help="Wind's presence in the environment", required=False)
    parser.add_argument('-no_w', '--no_wind', action="store_true", dest='windless', help="Wind's presence in the environment", required=False)
    parser.set_defaults(windless=False)
    parser.add_argument('-r', '--render', action="store_true", dest="render", help="Environment rendering at each step", required=False)
    parser.add_argument('-no_r', '--no_render', action="store_false", dest="render", help="No environment rendering at each step", required=False)
    parser.set_defaults(render=False)
    parser.add_argument('-csv', '--csv_filename', default="scores.csv", help="csv file to store scores in case of starting from a specific state", type=str, required=False)
    parser.add_argument('-k', '--length_k', default=5, help="Length of SXps", type=int, required=False)

    parser.add_argument('-spec_his', '--specific_history', default='', help="Express the specific history", type=str, required=False)
    parser.add_argument('-strat', '--strategy', default="exh", help="Type of strategy for (approximate) HXp",
                        type=str, required=False)
    parser.add_argument('-n', '--n', default=1, help="Most important action to highlight", type=int, required=False)
    parser.add_argument('-HXp', '--HXp', dest="HXP", action="store_true", help="Compute HXp", required=False)
    parser.add_argument('-no_HXp', '--no_HXp', action="store_false", dest="HXP", help="Do not compute HXp", required=False)
    parser.set_defaults(HXP=True)
    parser.add_argument('-pre', '--predicate', default="perfect cover", help="Predicate to verify in the history", type=str,
                        required=False)
    parser.add_argument('-imp_type', '--importance_type', default="action", help="To compute HXp without user queries, choice of the type of importance to search", type=str, required=False)
    parser.add_argument('-find_histories', '--find_histories', dest="find_histories", action="store_true", help="Find n histories", required=False)
    parser.add_argument('-no_find_histories', '--no_find_histories', action="store_false", dest="find_histories", help="Don't look for n histories", required=False)
    parser.set_defaults(find_histories=False)
    parser.add_argument('-strats', '--HXp_strategies', default="", help="Exploration strategies for similarity measures", type=str, required=False)
    parser.add_argument('-tr', '--transitions', default="approx",
                        help="Type of transitions at each succ step (exh / approx)", type=str, required=False)

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
    MAP_NAME = args.map_name
    NUMBER_EPISODES = args.nb_episodes
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUMBER_AGENTS = args.number_agents
    VIEW_RANGE = args.view_range
    WINDLESS = args.windless
    RANDOM_STARTING_POSITION = args.random_starting_position
    MOVE = args.step_move
    LIMIT = args.time_horizon
    K = args.length_k

    # only used if SPECIFIC_STATE
    CSV_FILENAME = args.csv_filename
    RENDER = args.render

    # used for HXp
    COMPUTE_HXP = args.HXP
    history_file = args.specific_history
    STRATEGY = args.strategy
    PREDICATE = args.predicate
    FIND_HISTORIES = args.find_histories
    IMPORTANCE_TYPE = args.importance_type
    N = args.n
    STRATEGIES = args.HXp_strategies
    TRANSITION = args.transitions

    backward = args.backward
    coefs = args.coefficents
    NB_SAMPLE = args.paxp_samples
    SELECT = args.backward_select

    FIXED_HORIZON = args.fixed_horizon
    
    
    if COMPUTE_HXP and history_file != "":
        history_file = 'Histories' + os.sep + history_file

    #  Fill pre-defined history
    SPECIFIC_HISTORY = []
    if COMPUTE_HXP and history_file != "":
        file = open(history_file, 'r')
        lines = file.readlines()
        cpt = 0

        for idx, line in enumerate(lines):
            if idx % 2:
                tmp_row = [int(t) for t in line[1:-2].split(', ')]
                SPECIFIC_HISTORY.append(tmp_row)
            else:
                tmp_row = [[int(t) for t in sublist.split(', ')] for sublist in line[2:-3].split('], [')]
                SPECIFIC_HISTORY.append(tmp_row)

    OSIRIM = False
    abs_dir_path = os.getcwd() + os.sep + 'DroneCoverage' if OSIRIM else os.getcwd()

    # Path to store actions utility
    if COMPUTE_HXP:
        utility_dirpath = abs_dir_path + os.sep + 'Utility' + os.sep + MAP_NAME
        if not os.path.exists(utility_dirpath):
            os.mkdir(utility_dirpath)
        utility_csv = utility_dirpath + os.sep + CSV_FILENAME

    else:
        utility_csv = 'scores.csv'

    # Path to store similarity measure
    if COMPUTE_HXP and FIND_HISTORIES:
        hist_dirpath = abs_dir_path + os.sep + 'Histories' + os.sep + str(NUMBER_EPISODES)+'-histories'
        if not os.path.exists(hist_dirpath):
            os.mkdir(hist_dirpath)
        hist_csv = hist_dirpath + os.sep + CSV_FILENAME

    else:
        hist_csv = 'trash.csv'

    #  Initialization --------------------------------------------------------------------------------------------------

    #  Env
    env = DroneAreaCoverage(map_name=MAP_NAME, windless=WINDLESS)

    #  Agents
    agents = []
    for i in range(NUMBER_AGENTS):
        agent = Agent(i + 1, env, view_range=VIEW_RANGE)
        agents.append(agent)

    env.initPos(agents, RANDOM_STARTING_POSITION)
    env.initObs(agents)

    #  Load net 
    net = DQN(np.array(agent.observation[0]).shape, np.array(agent.observation[1]).shape, agent.actions).to(DEVICE)
    net.load_state_dict(torch.load(abs_dir_path + os.sep + PATHFILE_MODEL, map_location=DEVICE))
    
    # Initialize HXP class
    if COMPUTE_HXP:
        predicates = {'perfect_cover': perfect_cover,
                      'imperfect_cover': imperfect_cover,
                      'no_drones': no_drones,
                      'crash': crash,
                      'max_reward': max_reward,
                      'region': region,
                      'redefined_predicate': redefined_predicate}
        functions = [transition, terminal, constraint, sample, preprocess, get_actions, render]

        # HXP use whether all transitions or only the ones of the agent to explain
        all_transitions = get_transition_exhaustive_lists(env, TRANSITION, len(agents))
        #print(all_transitions)

        add_info = {'agent': agents,
                    'net': net,
                    'history_transitions': [],
                    'all_transitions': all_transitions,
                    'select': SELECT,
                    'nb_sample': NB_SAMPLE,
                    'fixed_horizon': FIXED_HORIZON}

        hxp = HXP('DC', agents, env, predicates, functions, add_info, context='multi')

    # Convert horizon and delta type for backward HXP
    if backward:
        str_backward = backward.split(',')
        BACKWARD = [int(str_backward[0]), float(str_backward[1])]
        #print('BACKWARD: {}'.format(BACKWARD))

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
        # Clean environment
        env.clear_map()
        for agent in agents:
            agent.set_env(env)

        # Fill in the history
        specific_history = queue.Queue(maxsize=K * 2 + 1)
        for sa in SPECIFIC_HISTORY:  # specific_list
            specific_history.put(sa)

        # Render
        env_copy = deepcopy(env)
        agents_copy = deepcopy(agents)
        pos, actions = None, None
        for elm in SPECIFIC_HISTORY:
            #print(elm)
            if pos is None:
                pos = elm
                #print('pos ', pos)

            elif actions is None:
                actions= elm
                #print('actions ', actions)

            else:
                env_copy.init_map()
                env_copy.set_lastactions(actions)
                for i, s in enumerate(pos):
                    env_copy.set_initPos(agents_copy[i], s)
                env_copy.initObs(agents_copy)
                env_copy.render(agents_copy)
                pos = elm
                actions = None

        env_copy.init_map()
        env_copy.set_lastactions(None)

        for i, s in enumerate(pos):
            env_copy.set_initPos(agents_copy[i], s)

        env_copy.initObs(agents_copy)
        env_copy.render(agents_copy)

        # Compute (B-)HXP
        if not STRATEGIES:
            hxp.explain('no_user', specific_history, PREDICATE, [STRATEGY], N, IMPORTANCE_TYPE, utility_csv, BACKWARD, COEFS)
        else:
            hxp.explain('compare', specific_history, PREDICATE, STRATEGIES, N, IMPORTANCE_TYPE, utility_csv, BACKWARD, COEFS)

    #  Find histories
    elif FIND_HISTORIES and COMPUTE_HXP:
        env.reset(agents, rand=RANDOM_STARTING_POSITION)
        nb_scenarios = NUMBER_EPISODES
        storage = []
        pred_name, params = hxp.extract(PREDICATE)
        info = {'agent': agents, 'pred_params': params}
        print('Number of histories to find: {}'.format(nb_scenarios))

        # interaction loop
        while len(storage) != nb_scenarios:
            history = queue.Queue(maxsize=K * 2 + 1)
            #if not len(storage) % 50:
            #print('histories found: {}'.format(len(storage)))

            #  Reset env
            env.reset(agents, rand=RANDOM_STARTING_POSITION)
            done = False
            #  Store current agent's positions
            old_positions = [agent.get_obs()[1] for agent in agents]
            history.put(old_positions)
            cpt_stop = 0
            cpt = 0

            while cpt <= LIMIT:
                #  Choose action
                actions = []
                for agent in agents:
                    action = agent.predict(net, epsilon=0, device=DEVICE)
                    actions.append(action)

                #  History update
                if cpt_stop < 2:
                    if history.full():
                        history.get()
                        history.get()
                    history.put(actions)

                #  Step
                _, _, new_states, dones, _ = env.step(agents, actions, move=MOVE)
                #  Store current agent's positions
                positions = [agent.get_obs()[1] for agent in agents]
                #print(old_positions)

                if old_positions == positions:
                    cpt_stop += 1
                    #print('increase cpt stop')

                #  History update
                if cpt_stop < 2:
                    history.put(positions)
                    if valid_history(new_states, pred_name, info) and history.full():
                        data = [list(history.queue)]
                        storage.append(data)
                        print('progress bar: {}/{}'.format(len(storage), nb_scenarios))
                        break  # since we start from random configs, this will allow more diverse histories

                #  Check end of episode
                if dones.count(True) == len(dones) or dones[params[0] -1]:
                    break

                old_positions = positions
                cpt += 1

        # Store infos into CSV
        with open(hist_csv, 'a') as f:
            writer = csv.writer(f)
            # First Line
            line = ['History']
            writer.writerow(line)
            # Data
            for data in storage:
                writer.writerow(data)

    #  Classic testing loop
    else:
        dones = [False for _ in range(len(agents))]
        env.reset(agents, rand=RANDOM_STARTING_POSITION)
        cpt = 0
        env.render(agents)

        if HXP:
            history = queue.Queue(maxsize=K * 2 + 1)

        while not dones.count(True):
            env.reset(agents, rand=RANDOM_STARTING_POSITION)
            cpt = 0

            while cpt <= LIMIT:
                #  Choose action
                actions = []
                for agent in agents:
                    action = agent.predict(net, epsilon=0, device=DEVICE)
                    actions.append(action)
                #  Compute HXp
                if COMPUTE_HXP:
                    last_pos = [agent.get_obs()[1] for agent in agents]
                    history.put(last_pos)
                    # Compute HXp
                    if cpt and cpt >= K:
                        hxp.explain('user', history, approaches=[STRATEGY], n=N, imp_type=IMPORTANCE_TYPE, csv_file=utility_csv, backward=BACKWARD, coefs=COEFS)

                    # Update history
                    if history.full():
                        # Update history
                        history.get()
                        history.get()
                    history.put(actions)

                #  Step
                _, _, _, dones, _ = env.step(agents, actions, move=MOVE)
                #  Render

                env.render(agents)
                #  Extract rewards
                rewards = env.getReward(agents, actions, dones, reward_type="B")
                #  Check end of episode
                if dones.count(True) == len(dones) or last_pos == [agent.get_obs()[1] for agent in agents]:
                    break

                #  Display infos
                print("Dones True : {}".format(dones.count(True)))
                print("Rewards : {}".format(rewards))
                print("Cumulative reward : {}".format(sum(rewards)))
                print('-------')

                cpt += 1

        print('history:')
        length = 0
        for data in history.queue:
            print(data)
            length += 1
        print('length: {}'.format(length // 2))
