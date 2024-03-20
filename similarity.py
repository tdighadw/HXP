
import argparse
import csv
import os
import queue
import gymnasium as gym
import numpy as np
import torch
from scipy.spatial import distance
from math import sqrt
import inspect

# HXP
from HXP import HXP

#  FL
import FrozenLake.env as FL_env
import FrozenLake.agent as FL_agent
import FrozenLake.HXP_tools as FL_HXP_tools

#  DC
import DroneCoverage.env as DC_env
import DroneCoverage.agent as DC_agent
import DroneCoverage.DQN as DC_DQN
import DroneCoverage.HXP_tools as DC_HXP_tools

#  C4
import Connect4.env as C4_env
import Connect4.agent as C4_agent
import Connect4.DQN as C4_DQN
import Connect4.HXP_tools as C4_HXP_tools

#  DO
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.dqn import DQN as DO_DQN
import DynamicObstacles.HXP_tools as DO_HXP_tools

#  Parse a history of a problem
#  Input: history (str) and studied problem (str)
#  Output: state-action sequence, i.e. a history (state-action list)
def parse(str_history, problem):
    str_history = str_history[1:-1]

    # Parse String
    if problem == 'FL':
        # only one feature
        if not str_history.count('('):
            history = [int(i) for i in str_history.split(', ')]
        # several features
        else:
            # Create string history
            str_states_actions = str_history.split(', (')
            str_states = ['('+ sa.split(')')[0] + ')' if idx else sa.split(')')[0] + ')' for idx, sa in enumerate(str_states_actions)]
            str_actions = [sa.split('), ')[1] for sa in str_states_actions[:-1]]
            str_history = []

            for i in range(len(str_actions)):
                str_history.append(str_states[i])
                str_history.append(str_actions[i])
            str_history.append(str_states[-1])

            # Create history
            history = []
            for idx, line in enumerate(str_history):

                # state with several features
                if line[0] == '(':
                    tmp_row = tuple(int(t) for t in line[1:-1].split(', '))
                    history.append(tmp_row)
                # classic state / action
                else:
                    history.append(int(line[0]))

    elif problem == 'DC':
        #  Create string history
        str_states_actions = str_history.split('], [[')
        str_states = ['[['+ sa.split(']]')[0] + ']]' if idx else sa.split(']]')[0] + ']]' for idx, sa in enumerate(str_states_actions)]
        str_actions = [sa.split(']], ')[1] + ']' for sa in str_states_actions[:-1]]
        str_history = []

        for i in range(len(str_actions)):
            str_history.append(str_states[i])
            str_history.append(str_actions[i])
        str_history.append(str_states[-1])

        #  Create history
        history = []
        for idx, elm in enumerate(str_history):
            if idx % 2:
                tmp_row = [int(t) for t in elm[1:-1].split(', ')]
                history.append(tmp_row)
            else:
                tmp_row = [[int(t) for t in sublist.split(', ')] for sublist in elm[2:-2].split('], [')]
                history.append(tmp_row)

    elif problem in ['C4', 'DO']:
        open_br = '[[' if problem == 'C4' else '[[['
        close_br = ']]' if problem == 'C4' else ')]]'

        #  Create string history
        str_states_actions = str_history.split(', '+open_br)
        str_states = [open_br + sa.split(close_br)[0] + close_br if idx else sa.split(close_br)[0] + close_br for idx, sa in enumerate(str_states_actions)]
        str_actions = [sa[-1] for sa in str_states_actions[:-1]]
        str_history = []

        for i in range(len(str_actions)):
            str_history.append(str_states[i])
            str_history.append(str_actions[i])
        str_history.append(str_states[-1])

        #  Create history
        history = []
        for idx, elm in enumerate(str_history):
            if idx % 2:
                history.append(int(elm))
            else:
                if problem == 'C4':
                    tmp_row = [[int(t) for t in sublist.split(', ')] for sublist in elm[2:-2].split('], [')]

                elif problem == 'DO':
                    obs, pos = elm.split('], [(')
                    sublists = obs[2:-2].split(']], [[')
                    sublists[0] = sublists[0][2:]
                    tmp_obs = []
                    for sublist in sublists:
                        tmp_subl = [[int(t) for t in e.split(', ')] for e in sublist.split('], [')]
                        tmp_obs.append(tmp_subl)

                    tmp_pos = []
                    pos = pos.replace(']', '')
                    pos = pos.replace('[', '')
                    pos = pos[:-1]

                    for p in pos.split('), ('):
                        tmp_p = tuple(int(t) for t in p.split(', '))
                        tmp_pos.append(tmp_p)
                    tmp_pos = [tmp_pos[0], tmp_pos[1], [tmp_pos[2], tmp_pos[3]], tmp_pos[4]]
                    tmp_row = [tmp_obs,  tmp_pos]

                history.append(tmp_row)

    # Put the history in a queue
    queue_history = queue.Queue(maxsize=len(history))
    for elm in history:
        queue_history.put(elm)
    if problem == 'C4':
        return queue_history, history[0]
    else:
        return queue_history, None

#  Write the first line of a CSV file
#  Input: strategies used for the computation of action importance scores (str list)
#  Output: first line of the CSV file (str list)
def first_line(strats):
    line = ['History']

    for method in strats:
        line.append(method)
        line.append('time of '+method)

    for i in range(1, len(strats)):
        line.append(strats[0] + ' -- ' + strats[i])

    return line

if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', '--file', default="", help="History file", type=str, required=True)
    parser.add_argument('-new_file', '--new_file', default="", help="Store history, importance scores and time", type=str, required=True)
    parser.add_argument('-pre', '--predicate', default="", help="predicate to verify", type=str, required=True)
    parser.add_argument('-k', '--k', default=5, help="Scenarios length", type=int, required=False)
    parser.add_argument('-pre_info', '--predicate_additional_info', default=None, help="Specify a state", type=str, required=False)
    parser.add_argument('-problem', '--problem', default="", help="considered problem", type=str, required=True)
    parser.add_argument('-strats', '--HXp_strategies', default="[exh, last_1, last_2, last_3, last_4, transition_1, transition_2, transition_3, transition_4]",
                        help="Exploration strategies for similarity measures", type=str, required=False)
    parser.add_argument('-policy', '--policy', default="", help="Policy name/file", type=str,
                        required=True)
    parser.add_argument('-map', '--map_name', default="10x10", help="Map's dimension (nxn)", type=str, required=False)
    parser.add_argument('-agents', '--number_agents', default=4, help="Number of agents in the map", type=int,
                        required=False)
    parser.add_argument('-tr', '--transitions', default="approx",
                        help="Type of transitions at each succ step (exh / approx)", type=str, required=False)
    parser.add_argument('-features', '--many_features', dest="many_features", action="store_true",
                        help="Several features to define agent's state", required=False)
    parser.add_argument('-no_features', '--no_many_features', action="store_false", dest="many_features",
                        help="Only one feature to define the agent's state", required=False)
    parser.set_defaults(many_features=False)

    args = parser.parse_args()

    # Get arguments
    FILE = args.file
    NEW_FILE = args.new_file
    K = args.k
    PREDICATE = args.predicate
    ADD_INFO = args.predicate_additional_info
    PROBLEM = args.problem
    STRATEGIES = args.HXp_strategies.split(', ')
    STRATEGIES[0] = STRATEGIES[0][1:]
    STRATEGIES[-1] = STRATEGIES[-1][:-1]
    POLICY = args.policy
    # FL-DO argument(s)
    MAP = args.map_name
    FEATURES = args.many_features
    # DC argument(s)
    NUMBER_AGENTS = args.number_agents
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TRANSITION = args.transitions

    abs_dir_path = os.getcwd()

    # Initialize agent(s) and environment
    if PROBLEM == 'FL':

        # Convert additional information
        if ADD_INFO is not None:
            if ADD_INFO[0] == '[':
                ADD_INFO = [int(i) for i in ADD_INFO[1:-1].split(', ')]
            else:
                ADD_INFO = int(ADD_INFO)

        agent_Q_dirpath = "Q-tables" + os.sep + "Agent"
        #  Env initialization
        env = FL_env.MyFrozenLake(map_name=MAP, slip_probas=[0.2, 0.6, 0.2], many_features=FEATURES)
        #  Agent initialization
        agent = FL_agent.Agent(POLICY, env)
        #  Load Q table
        agent.load(agent_Q_dirpath)
        # Initialize HXP class
        predicates = {'win': FL_HXP_tools.win,
                    'holes': FL_HXP_tools.holes,
                    'specific_state': FL_HXP_tools.specific_state,
                    'specific_part': FL_HXP_tools.specific_part,
                    'redefined_predicate': FL_HXP_tools.redefined_predicate}
        functions = [FL_HXP_tools.transition,
                     FL_HXP_tools.terminal,
                     FL_HXP_tools.constraint,
                     FL_HXP_tools.sample,
                     FL_HXP_tools.preprocess,
                     FL_HXP_tools.get_actions,
                     FL_HXP_tools.render]
        add_info = {'select': 'imp', 'nb_sample': -1, 'fixed_horizon': True}
        hxp = HXP('FL', agent, env, predicates, functions, add_info)

    elif PROBLEM == 'DC':

        #  Env initialization
        env = DC_env.DroneAreaCoverage(map_name=MAP)
        #  Agent initialization
        agents = []
        for i in range(NUMBER_AGENTS):
            agent = DC_agent.Agent(i + 1, env)
            agents.append(agent)
        env.initPos(agents, True)
        env.initObs(agents)
        #  Net loading
        net = DC_DQN.DQN(np.array(agent.observation[0]).shape, np.array(agent.observation[1]).shape, agent.actions).to(DEVICE)
        net.load_state_dict(torch.load(abs_dir_path + POLICY, map_location=DEVICE))
        # Initialize HXP class TODO
        predicates = {'perfect_cover': DC_HXP_tools.perfect_cover,
                      'imperfect_cover': DC_HXP_tools.imperfect_cover,
                      'no_drones': DC_HXP_tools.no_drones,
                      'crash': DC_HXP_tools.crash,
                      'max_reward': DC_HXP_tools.max_reward,
                      'region': DC_HXP_tools.region,
                      'redefined_predicate': DC_HXP_tools.redefined_predicate}
        functions = [DC_HXP_tools.transition,
                     DC_HXP_tools.terminal,
                     DC_HXP_tools.constraint,
                     DC_HXP_tools.sample,
                     DC_HXP_tools.preprocess,
                     DC_HXP_tools.get_actions,
                     DC_HXP_tools.render]
        all_transitions = DC_HXP_tools.get_transition_exhaustive_lists(env, TRANSITION, len(agents))
        add_info = {'agent': agents,
                    'net': net,
                    'history_transitions': [],
                    'all_transitions': all_transitions,
                    'select': 'imp',
                    'nb_sample': -1,
                    'fixed_horizon': True}
        hxp = HXP('DC', agents, env, predicates, functions, add_info, context='multi')

    elif PROBLEM == 'C4':

        #  Env initialization
        env = C4_env.Connect4()
        #  Agents initialization
        player_1 = C4_agent.Agent('Yellow', env)
        player_2 = C4_agent.Agent('Red', env)
        #  Net Loading
        net = C4_DQN.DQN((env.rows, env.cols), env.action_space.n).to(DEVICE)
        net.load_state_dict(torch.load(abs_dir_path + POLICY, map_location=DEVICE))
        # Initialize HXP class
        predicates = {'win': C4_HXP_tools.win,
                      'lose': C4_HXP_tools.lose,
                      'control_midcolumn':  C4_HXP_tools.control_midcolumn,
                      'align3': C4_HXP_tools.align3,
                      'counteralign3': C4_HXP_tools.counteralign3,
                      'redefined_predicate': C4_HXP_tools.redefined_predicate}
        functions = [C4_HXP_tools.transition,
                     C4_HXP_tools.terminal,
                     C4_HXP_tools.constraint,
                     C4_HXP_tools.sample,
                     C4_HXP_tools.preprocess,
                     C4_HXP_tools.get_actions,
                     C4_HXP_tools.render]
        add_info = {'opponent': player_2,
                    'net': net,
                    'initial_s': None,
                    'select': 'imp',
                    'nb_sample': -1,
                    'fixed_horizon': True}
        hxp = HXP('C4', player_1, env, predicates, functions, add_info)

    elif PROBLEM == 'DO':

        #  Env initialization
        env = gym.make(MAP)
        env = ImgObsWrapper(env)
        _, _ = env.reset()
        #  Net loading
        model = DO_DQN.load(abs_dir_path + POLICY)
        # Initialize HXP class
        predicates = {'success': DO_HXP_tools.success,
                      'collision': DO_HXP_tools.collision,
                      'specific_position': DO_HXP_tools.specific_position,
                      'avoid_specific_position': DO_HXP_tools.avoid_specific_position,
                      'close_balls': DO_HXP_tools.close_balls,
                      'redefined_predicate': DO_HXP_tools.redefined_predicate}
        functions = [DO_HXP_tools.transition,
                     DO_HXP_tools.terminal,
                     DO_HXP_tools.constraint,
                     DO_HXP_tools.sample,
                     DO_HXP_tools.preprocess,
                     DO_HXP_tools.get_actions,
                     DO_HXP_tools.render]
        add_info = {'net': model, 'env': env, 'select': 'imp', 'nb_sample': -1, 'fixed_horizon': True}
        hxp = HXP('DO', model, env, predicates, functions, add_info)

    # specific to 'similarity.py'
    # Read first cell of all lines to get the History
    with open(abs_dir_path + FILE, 'r') as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            print(row)
            if row[0] not in ['History', '']:
                d = []
                history, initial_s = parse(row[0], PROBLEM)
                #print('History is: ', history)

                # Run HXp for each strategy
                if PREDICATE in ['control_midcolumn', 'align3', 'counteralign3'] and PROBLEM == 'C4':
                    hxp.add_info['initial_s'] = initial_s
                _, scores, times, _ = hxp.explain('compare', history, PREDICATE, STRATEGIES)

                # Store data (history - strategy imp score - strategy computational time)
                d.append(list(history.queue))
                for i in range(len(scores)):
                    d.append(scores[i])
                    d.append(times[i])

                # Compute and store data (similarity scores)
                for i in range(1, len(scores)):
                    d.append(1 - (distance.euclidean(scores[0], scores[i]) / 2 * sqrt(K)))
                data.append(d)

    # Write stored data
    with open(abs_dir_path + os.sep + NEW_FILE, 'a') as f:
        writer = csv.writer(f)
        # First Line
        line = first_line(STRATEGIES)
        writer.writerow(line)

        # Data
        for d in data:
            writer.writerow(d)

        # Means & Std
        index = 2
        line_avg = [''] * index
        line_std = [''] * index

            # Time
        for i in range(index, len(data[0]) - len(STRATEGIES) + 2, 2):
            times = [d[i] for d in data]
            avg = np.mean(times, axis=0)
            std = np.std(times, axis=0)
            if i != len(data[0]) - len(STRATEGIES):
                line_avg.extend([avg, ''])
                line_std.extend([std, ''])
            else:
                line_avg.extend([avg])
                line_std.extend([std])

            # Similarity
        for i in range(len(data[0]) - len(STRATEGIES) + 1, len(data[0])):
            sim = [d[i] for d in data]
            avg = np.mean(sim, axis=0)
            std = np.std(sim, axis=0)
            line_avg.extend([avg])
            line_std.extend([std])
        writer.writerow(line_avg)
        writer.writerow(line_std)
