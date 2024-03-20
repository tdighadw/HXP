import argparse
import csv
import os
import queue
import sys
from minigrid.wrappers import *
from stable_baselines3.dqn import DQN
from HXP_tools import get_positions, hxp_render

# Get access to the HXP file
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from HXP import HXP
from HXP_tools import success, collision, close_balls, specific_position, avoid_specific_position, redefined_predicate
from HXP_tools import transition, terminal, render, preprocess, get_actions, constraint, sample
from HXP_tools import valid_history

#  From np array to list
#  Input: observation (np array)
#  Output: observation (list)
def from_np_to_list(obs):
    obs_list = []
    for line in obs:
        line_list = []
        for cell in line:
            line_list.append(list(cell))
        obs_list.append(line_list)
    return obs_list

if __name__ == "__main__":

    #  Parser ----------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('-ep', '--nb_episodes', default=1,
                        help="Number of episodes for a classic test of agent's policy", type=int, required=False)
    #  Directory Paths
    parser.add_argument('-model', '--model_dir', default="Models", help="Agent's model directory", type=str, required=False)
    #  Policy name (Best policy: 2M-Wed_Nov_11__11:11:46__AM_2023 | Policy used for experiments: 300k-Mon_Jan_01__06:43:10__PM_2024)
    parser.add_argument('-policy', '--policy_name', default="2M-Wed_Nov_11__11:11:46__AM_2023", help="Filename of the agent's policy", type=str, required=False)
    # Env
    parser.add_argument('-map', '--map_name', default="MiniGrid-Dynamic-Obstacles-5x5-v0", help="Map's name", type=str, required=False)

    parser.add_argument('-r', '--render', action="store_true", dest="render", help="Environment rendering at each step", required=False)
    parser.add_argument('-no_r', '--no_render', action="store_false", dest="render", help="No environment rendering at each step", required=False)
    parser.set_defaults(render=False)
    # HXP parameters
    parser.add_argument('-n', '--n', default=1, help="Most important action to highlight", type=int, required=False)
    parser.add_argument('-k', '--length_k', default=5, help="Length of HXPs", type=int, required=False)
    parser.add_argument('-spec_his', '--specific_history', default='', help="Express the specific history", type=str,
                        required=False)
    parser.add_argument('-pre', '--predicate', default="success", help="Predicate to verify in the history",
                        type=str,
                        required=False)
    parser.add_argument('-strat', '--strategy', default="exh", help="Type of strategy for (approximate) HXp",
                        type=str, required=False)
    parser.add_argument('-imp_type', '--importance_type', default="action", help="To compute HXp without user queries, choice of the type of importance to search", type=str, required=False)


    parser.add_argument('-csv', '--csv_filename', default="scores.csv",
                        help="csv file to store scores in case of starting from a specific state", type=str,
                        required=False)
    parser.add_argument('-strats', '--HXp_strategies', default="",
                        help="Exploration strategies for similarity measures", type=str, required=False)
    parser.add_argument('-backward', '--backward', default='', help="Specify (max) horizon and delta for backward HXP",
                        type=str, required=False)
    parser.add_argument('-coefs', '--coefficents', default='',
                        help="coefficients for B-HXP computation with two studied predicates",
                        type=str, required=False)

    parser.add_argument('-HXp', '--HXp', dest="HXP", action="store_true", help="Compute HXp", required=False)
    parser.add_argument('-no_HXp', '--no_HXp', action="store_false", dest="HXP", help="Do not compute HXp", required=False)
    parser.set_defaults(HXP=True)
    parser.add_argument('-find_histories', '--find_histories', dest="find_histories", action="store_true", help="Find n histories", required=False)
    parser.add_argument('-no_find_histories', '--no_find_histories', action="store_false", dest="find_histories", help="Don't look for n histories", required=False)
    parser.set_defaults(find_histories=False)

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
    NUMBER_EPISODES = args.nb_episodes

    MODEL_DIR = args.model_dir

    POLICY = args.policy_name

    MAP = args.map_name
    RENDER = args.render

    N = args.n
    K = args.length_k
    history_file = args.specific_history
    PREDICATE = args.predicate
    STRATEGY = args.strategy
    IMPORTANCE_TYPE = args.importance_type
    CSV_FILENAME = args.csv_filename
    STRATEGIES = args.HXp_strategies

    backward = args.backward
    coefs = args.coefficents
    NB_SAMPLE = args.paxp_samples
    SELECT = args.backward_select

    COMPUTE_HXP = args.HXP
    FIND_HISTORIES = args.find_histories

    FIXED_HORIZON = args.fixed_horizon

    # Get the history --------------------------------------------------------------------------------------------------

    SPECIFIC_HISTORY = []
    if COMPUTE_HXP and history_file != "":
        # Access to the history file
        history_file = 'Histories' + os.sep + history_file
        #  Fill pre-defined history
        file = open(history_file, 'r')
        lines = file.readlines()
        state = []
        feature = []
        cpt = 0

        for idx, line in enumerate(lines):
            split_line = [st for st in line.split(' ') if st not in ['', '\n']]
            if split_line:
                # Case: state
                if split_line[0].startswith('['):
                    feature.append([int(split_line[0][-1]), int(split_line[1]), int(split_line[-1][0])])
                    cpt_bracket = list(split_line[-1]).count(']')
                    # Case: end of a feature
                    if cpt_bracket >= 2:
                        state.append(feature)
                        feature = []

                # Case: environment variables
                elif split_line[0].startswith('('):
                    env_variables = []
                    obstacles = False
                    for idx, elm in enumerate(split_line):

                        # first part of the tuple
                        if not idx % 2:
                            tmp_tuple = []
                            tmp_tuple.append(int(elm[-2]))
                            if elm[0] == '[':
                                obstacles = True
                                tmp_obs_pos = []

                        # second part
                        else:
                            tmp_tuple.append(int(elm[0]))
                            if not obstacles:
                                env_variables.append(tuple(tmp_tuple))
                            else:
                                tmp_obs_pos.append(tuple(tmp_tuple))
                                if len(tmp_obs_pos) == 2:
                                    env_variables.append(tmp_obs_pos)
                                    obstacles = False

                    SPECIFIC_HISTORY.append([state, env_variables])
                    state = []

                # Case: action
                else:
                    SPECIFIC_HISTORY.append(int(split_line[0][0]))
    # Display history
    for elm in SPECIFIC_HISTORY:
        print(elm)

    print("-------------")

    # Path to store actions utility
    if COMPUTE_HXP:
        utility_dirpath = 'Utility' + os.sep + MAP
        if not os.path.exists(utility_dirpath):
            os.mkdir(utility_dirpath)
        utility_csv = utility_dirpath + os.sep + CSV_FILENAME
    else:
        utility_csv = 'scores.csv'

    # Path to store similarity measure
    if COMPUTE_HXP and FIND_HISTORIES:
        hist_dirpath = 'Histories' + os.sep + str(NUMBER_EPISODES)+'-histories'
        if not os.path.exists(hist_dirpath):
            os.mkdir(hist_dirpath)
        hist_csv = hist_dirpath + os.sep + CSV_FILENAME
    else:
        hist_csv = 'trash.csv'

    #  Initialization --------------------------------------------------------------------------------------------------

    # Env
    human_render = 'human' if RENDER else 'rgb_array'
    env = gym.make(MAP, render_mode=human_render)
    env = ImgObsWrapper(env)
    obs, _ = env.reset()
    #env.render()

    # Load net
    model = DQN.load(MODEL_DIR + os.sep + POLICY)

    # Initialize HXP class
    if COMPUTE_HXP:
        predicates = {'success': success,
                      'collision': collision,
                      'close_balls': close_balls,
                      'specific_position': specific_position,
                      'avoid_specific_position': avoid_specific_position,
                      'redefined_predicate': redefined_predicate}
        functions = [transition, terminal, constraint, sample, preprocess, get_actions, render]
        add_info = {'net': model, 'env': env, 'select': SELECT, 'nb_sample': NB_SAMPLE, 'fixed_horizon': FIXED_HORIZON}
        hxp = HXP('DO', model, env, predicates, functions, add_info)

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
        # Fill in the history
        specific_history = queue.Queue(maxsize=K * 2 + 1)
        i = 0
        for sa in SPECIFIC_HISTORY:  # specific_list
            if not i or not i % 2:
                hxp_render(sa[0])
            else:
                print(sa)
            specific_history.put(sa)
            i += 1

        # Compute HXp
        if not STRATEGIES:
            hxp.explain('no_user', specific_history, PREDICATE, [STRATEGY], N, IMPORTANCE_TYPE, utility_csv, BACKWARD, COEFS)
        else:
            hxp.explain('compare', specific_history, PREDICATE, STRATEGIES, N, IMPORTANCE_TYPE, utility_csv, BACKWARD, COEFS)

    # Find histories
    elif FIND_HISTORIES and COMPUTE_HXP:
        nb_scenarios = NUMBER_EPISODES
        storage = []
        name, pred_params = hxp.extract(PREDICATE)
        info = {'env': env, 'pred_params': pred_params}

        # interaction loop
        while len(storage) != nb_scenarios:
            history = queue.Queue(maxsize=K * 2 + 1)
            obs, _ = env.reset()
            done = False
            #print('get_positions;', get_positions(env))
            obs_pos = [obs.tolist(), list(get_positions(env))]
            history.put(obs_pos)  # initial state

            while not done:
                action, _ = model.predict(obs, deterministic=True)

                if history.full():
                    history.get()
                    history.get()
                history.put(int(action))

                obs, reward, done, _, _ = env.step(action)
                obs_pos = [obs.tolist(), list(get_positions(env))]
                history.put(obs_pos)

                if valid_history(obs_pos, name, info) and history.full():
                    data = [list(history.queue)]
                    storage.append(data)
                    if len(storage) == nb_scenarios:  # deal with specific_cell predicate (more than 1 history per episode)
                        break

        # Store infos into CSV
        with open(hist_csv, 'a') as f:
            writer = csv.writer(f)
            # First Line
            line = ['History']
            writer.writerow(line)
            # Data
            for data in storage:
                #print(data)
                writer.writerow(data)

    # Classic testing loop
    else:
        nb_episodes = NUMBER_EPISODES
        cpt = 0
        while cpt != 12:
            print('New episode')
            cpt = 0
            done = False
            env.reset()
            env.render()
            #time.sleep(5)
            if COMPUTE_HXP:
                history = queue.Queue(maxsize=K * 2 + 1)
            while not done:
                print(obs)
                print(get_positions(env))

                #  Choose action
                action, _ = model.predict(obs, deterministic=True)
                #print(action)

                #  Compute HXp
                if COMPUTE_HXP:
                    history.put([from_np_to_list(obs), list(get_positions(env))])
                    # Compute (B-)HXp
                    if cpt and cpt >= K:
                        print(cpt)
                        hxp.explain('user', history, approaches=[STRATEGY], n=N, imp_type=IMPORTANCE_TYPE, csv_file=utility_csv, backward=BACKWARD, coefs=COEFS)
                    # Update history
                    if history.full():
                        # Update history
                        history.get()
                        history.get()
                    history.put(action)
                #  Step
                obs, reward, done, _, _ = env.step(action)
                #  Render
                env.render()

                cpt += 1

            print(obs)
            print(get_positions(env))
            print("Final Reward: {}".format(reward))
            print('length: {}'.format(cpt))

            # Last (B-)HXP
            if COMPUTE_HXP and cpt >= K:
                history.put([from_np_to_list(obs), list(get_positions(env))])
                hxp.explain('user', history, approaches=[STRATEGY], n=N, imp_type=IMPORTANCE_TYPE, csv_file=utility_csv, backward=BACKWARD, coefs=COEFS)
