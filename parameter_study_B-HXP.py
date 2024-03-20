import argparse
import csv
import os
import queue
import sys

import numpy as np
import torch
from HXP import HXP
import gymnasium as gym

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
#  Output: state-action sequence, i.e. a history (state-action list), first history state (state
def parse(str_history, problem):
    str_history = str_history[1:-1]

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
#  Input: hyper-parameter under study (str) and the of studied values (float/int)
#  Output: first line of the CSV file (str list)
def first_line(hyper_param, values):
    line = ['History']

    for v in values:
        # score + predicate
        line.append(hyper_param + ' = ' + v)
        line.append('')

        # metrics for comparison
        line.append('time (' + hyper_param + ' = ' + v + ')')
        line.append('action sparsity (' + hyper_param + ' = ' + v + ')')
        line.append('mean predicate sparsity (' + hyper_param + ' = ' + v + ')')

    return line

#  Compute the predicate sparsity score, given a RL problem and specific predicate.
#  The predicate is defined during the B-HXP computation.
#  Input: RL problem considered (str), predicate (state)
#  Output: predicate sparsity score (float)
def predicate_sparsity(problem, pred):
    # get total number of features / number of fixed features
    if problem == 'FL':
        total = len(pred)
        fixed_features = total - pred.count(None)

    elif problem in ['DO', 'C4']:
        total = len(pred) * len(pred[0])
        fixed_features = sum([len(line) - line.count(None) for line in pred])

    elif problem == 'DC':
        total_view = len(pred[0]) * len(pred[0][0])
        total_position = len(pred[1])
        total = total_view + total_position
        fixed_view = total_view - sum([len(line) - line.count(None) for line in pred[0]])
        fixed_position =  total_position - pred[1].count(None)
        fixed_features = fixed_view + fixed_position

    else:
        return None

    # compute predicate sparsity score
    return min(1 - ((fixed_features - 1) / (total - 1)), 1.0)

#  Change the parameter value
#  Input: parameter (str), value (int/float), hxp instance (HXP), current values of l and delta (int/float list)
#  Output: values of l and delta (int/float list), hxp instance (HXP)
def modify_value(parameter, value, hxp, backward):
    if parameter == 'sample':
        hxp.add_info['nb_sample'] = int(value)
    elif parameter == 'l':
        backward[0] = int(value)
    elif parameter == 'delta':
        backward[1] = float(value)
    return backward, hxp

if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', '--file', default="", help="History file", type=str, required=True)
    parser.add_argument('-new_file', '--new_file', default="", help="Store history, importance scores and time", type=str, required=True)
    parser.add_argument('-pre', '--predicate', default="", help="predicate to verify", type=str, required=True)
    parser.add_argument('-k', '--k', default=10, help="Scenarios length", type=int, required=False)
    parser.add_argument('-pre_info', '--predicate_additional_info', default=None, help="Specify a state", type=str, required=False)
    parser.add_argument('-problem', '--problem', default="", help="considered problem", type=str, required=True)
    parser.add_argument('-strats', '--HXp_strategies', default="[exh]",
                        help="Exploration strategies for similarity measures", type=str, required=False)
    parser.add_argument('-param', '--parameter', default="delta",
                        help="Parameter to study", type=str, required=False)
    parser.add_argument('-values', '--parameter_values', default="",
                        help="Values to test for the studied parameter", type=str, required=True)
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
    parser.add_argument('-samples', '--paxp_samples', default=-1,
                        help="Maximal number of state samples for the evaluation of a feature during the PAXp computation", type=int, required=False)

    parser.add_argument('-backward', '--backward', default='', help="Specify (max) horizon and delta for backward HXP",
                        type=str, required=False)

    args = parser.parse_args()

    # Get arguments
    FILE = args.file
    NEW_FILE = args.new_file
    K = args.k
    PREDICATE = args.predicate
    ADD_INFO = args.predicate_additional_info
    PROBLEM = args.problem
    STRATEGIES = args.HXp_strategies.split(', ') if ',' in args.HXp_strategies else [args.HXp_strategies]
    STRATEGIES[0] = STRATEGIES[0][1:]
    STRATEGIES[-1] = STRATEGIES[-1][:-1]
    POLICY = args.policy
    PARAMETER = args.parameter
    VALUES = args.parameter_values.split(', ') if ',' in args.parameter_values else [args.parameter_values]
    VALUES[0] = VALUES[0][1:]
    VALUES[-1] = VALUES[-1][:-1]
    NB_SAMPLE = args.paxp_samples

    # FL-DO argument(s)
    MAP = args.map_name
    FEATURES = args.many_features
    # DC argument(s)
    NUMBER_AGENTS = args.number_agents
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TRANSITION = args.transitions

    abs_dir_path = os.getcwd()

    backward = args.backward
    str_backward = backward.split(',')
    BACKWARD = [int(str_backward[0]), float(str_backward[1])]
    print('BACKWARD: {}'.format(BACKWARD))

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
        add_info = {'select': 'imp', 'nb_sample': NB_SAMPLE, 'fixed_horizon': True}
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
        # Initialize HXP class
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
                    'nb_sample': NB_SAMPLE,
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
                    'nb_sample': NB_SAMPLE,
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
        add_info = {'net': model, 'env': env, 'select': 'imp', 'nb_sample': NB_SAMPLE, 'fixed_horizon': True}
        hxp = HXP('DO', model, env, predicates, functions, add_info)

    # Compute (B-)HXP for each history in FILE and store the results
    with open(abs_dir_path + os.sep + FILE, 'r') as f:
        reader = csv.reader(f)
        data = []
        i = 1
        for row in reader:

            print(row)
            if row[0] not in ['History', '']:
                d = []
                history, initial_s = parse(row[0], PROBLEM)
                d.append(list(history.queue))
                # print('History is: ', history)

                if PREDICATE in ['control_midcolumn', 'align3', 'counteralign3'] and PROBLEM == 'C4':
                    hxp.add_info['initial_s'] = initial_s

                # Run (B-)HXp for each strategy
                for v in VALUES:
                    BACKWARD, hxp = modify_value(PARAMETER, v, hxp, BACKWARD)
                    # print('backward: {}'.format(BACKWARD))
                    # print('sample: {}'.format(hxp.add_info['nb_sample']))

                    explanation, scores, times, predicates = hxp.explain('no_user', history, PREDICATE, STRATEGIES, backward=BACKWARD)
                    if hxp.add_info.get('redefined_predicate'): del hxp.add_info['redefined_predicate']

                    # Store data
                    d.append(scores)
                    d.append(predicates)
                    d.append(times[0])

                    # Compute action / predicate sparsity scores
                    action_sparsity = min(1 - ((len(explanation) - 1) / (K-1)), 1.0)
                    predicates_sparsity = []
                    for pred in predicates[:-1]:
                        predicates_sparsity.append(predicate_sparsity(PROBLEM, pred))
                    mean_predicate_sparsity = sum(predicates_sparsity) / len(predicates_sparsity) if predicates_sparsity else 1.0

                    # Store scores
                    d.append(action_sparsity)
                    d.append(mean_predicate_sparsity)

                data.append(d)
                print('--------------------------------')
                print('End of history: {}'.format(i))
                i += 1
                print('Size of data: {}'.format(sys.getsizeof(data)))
                print('--------------------------------')

    # Write stored results
    with open(abs_dir_path + os.sep + NEW_FILE, 'a') as f:
        writer = csv.writer(f)
        # First Line
        line = first_line(PARAMETER, VALUES)
        print('first line: ', line)
        writer.writerow(line)

        # Data
        for d in data:
            writer.writerow(d)

        # Means & Std
        index = 3
        line_avg = [''] * index
        line_std = [''] * index

        # Time / Sparsity action / Sparsity predicate
        for i in range(index, len(data[0]), 5): # - len(VALUES) + 5
            print('i:', i)
            times = [d[i] for d in data]
            avg_time = np.mean(times, axis=0)
            std_time = np.std(times, axis=0)

            sparsity_actions = [d[i + 1] for d in data]
            avg_sp_actions = np.mean(sparsity_actions, axis=0)
            std_sp_actions = np.std(sparsity_actions, axis=0)

            sparsity_predicates = [d[i + 2] for d in data]
            avg_sp_predicates = np.mean(sparsity_predicates, axis=0)
            std_sp_predicates = np.std(sparsity_predicates, axis=0)

            line_avg.extend([avg_time, avg_sp_actions, avg_sp_predicates])
            line_std.extend([std_time, std_sp_actions, std_sp_predicates])
            if i != len(data[0]) - len(VALUES):
                line_avg.extend(['', ''])
                line_std.extend(['', ''])

        print('Avg: {}'.format(line_avg))
        writer.writerow(line_avg)
        writer.writerow(line_std)
