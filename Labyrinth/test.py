import argparse
import json
import os
import queue
import sys

# Get access to the HXP file
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Labyrinth.env import LabrinthEnv
from HXP import HXP
from HXP_tools import reach_exit
from HXP_tools import transition, terminal, render, preprocess, get_actions, constraint, sample


if __name__ == "__main__":
    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-map', '--map_name', default="corridor", help="Map's dimension (nxn)", type=str,
                        required=False)
    parser.add_argument('-policy', '--policy_name', default="corridor_det_env", help="Common part of policy name",
                        type=str, required=False)
    parser.add_argument('-ep', '--nb_episodes', default=1, help="Number of testing episodes", type=int,
                        required=False)

    parser.add_argument('-stoch', '--stoch_env', dest="stochastic_env", action="store_true", help="Stochastic environment", required=False)
    parser.add_argument('-no_stoch', '--no_stoch_env', action="store_false", dest="stochastic_env", help="Deterministic environment", required=False)
    parser.set_defaults(stochastic_env=False)

    parser.add_argument('-HXp', '--HXp', dest="COMPUTE_HXP", action="store_true", help="Compute HXp", required=False)
    parser.add_argument('-no_HXp', '--no_HXp', action="store_false", dest="COMPUTE_HXP", help="Do not compute HXp", required=False)
    parser.set_defaults(COMPUTE_HXP=True)

    parser.add_argument('-imp_type', '--importance_type', default="action", help="To compute HXp without user queries, choice of the type of importance to search", type=str, required=False)
    parser.add_argument('-pre', '--predicate', default="reach_exit", help="Predicate to verify in the history", type=str, required=False)
    parser.add_argument('-n', '--n', default=1, help="Most important action to highlight", type=int, required=False)
    parser.add_argument('-spec_his', '--specific_history', default='', help="Express the specific history", type=str, required=False)
    parser.add_argument('-strat', '--strategy', default="exh", help="Exploration strategy for generating HXp", type=str, required=False)
    parser.add_argument('-strats', '--strategies', default="", help="Exploration strategies for similarity measures", type=str, required=False)
    parser.add_argument('-csv', '--csv_filename', default="scores.csv", help="csv file to store utility from an HXp", type=str, required=False)
    parser.add_argument('-k', '--length_k', default=5, help="Length of History", type=int, required=False)
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
    NB_EPISODES = args.nb_episodes
    STOCH_ENV = args.stochastic_env

    K = args.length_k
    STRATEGY = args.strategy
    COMPUTE_HXP = args.COMPUTE_HXP
    PREDICATE = args.predicate
    N = args.n
    IMPORTANCE_TYPE = args.importance_type
    STRATEGIES = args.strategies
    CSV_FILENAME = args.csv_filename
    SELECT = args.backward_select
    FIXED_HORIZON = args.fixed_horizon

    history_file = args.specific_history

    #  Fill the specific history list (convert string into int list)
    SPECIFIC_HISTORY = []
    if COMPUTE_HXP and history_file != "":
        history_file = 'Histories' + os.sep + history_file
        file = open(history_file, 'r')
        lines = file.readlines()
        cpt = 0

        for idx, line in enumerate(lines):
            SPECIFIC_HISTORY.append(int(line))
        print("Specific history : {}".format(SPECIFIC_HISTORY))

    # Path to store policy
    policy_dirpath = "Policies"

    # Path to store actions utility in case of HXp
    if COMPUTE_HXP:
        utility_dirpath = 'Utility' + os.sep + MAP_NAME
        if not os.path.exists(utility_dirpath):
            os.mkdir(utility_dirpath)
        utility_csv = utility_dirpath + os.sep + CSV_FILENAME

    else:
        utility_csv = 'scores.csv'

    # Init environment
    if STOCH_ENV:
        env = LabrinthEnv(map_name=MAP_NAME, proba=[0.5, 0.5])
    else:
        env = LabrinthEnv(map_name=MAP_NAME)

    # Get trained policy
    with open(policy_dirpath + os.sep + 'policy_' + POLICY_NAME, 'r') as fp:
        policy_list = json.load(fp)
        policy = {s: action for s, action in policy_list}

    # Initialize HXP class
    if COMPUTE_HXP:
        predicates = {'reach_exit': reach_exit}
        functions = [transition, terminal, constraint, sample, preprocess, get_actions, render]
        add_info = {'env': env, 'select': SELECT, 'fixed_horizon': FIXED_HORIZON}
        hxp = HXP('L', policy, env, predicates, functions, add_info)

    print(policy)
    # Compute HXp from a specific history
    if SPECIFIC_HISTORY and COMPUTE_HXP:
        specific_history = queue.Queue(maxsize=K * 2 + 1)
        for sa in SPECIFIC_HISTORY:  # specific_list
            specific_history.put(sa)

        if not STRATEGIES:
            hxp.explain('no_user', specific_history, PREDICATE, [STRATEGY], N, IMPORTANCE_TYPE, utility_csv)
        else:
            hxp.explain('compare', specific_history, PREDICATE, STRATEGIES, N, IMPORTANCE_TYPE, utility_csv)

    else:
        # Classic test loop to visualize agent's behavior
        for i in range(NB_EPISODES):
            # Reset env and done
            env.reset()
            done = False
            cpt = 0
            if COMPUTE_HXP:
                history = queue.Queue(maxsize=K * 2 + 1)
            while not done:
                action = policy[env.s]

                #  Compute HXp
                if COMPUTE_HXP:
                    history.put(env.s)
                    # Compute (B-)HXp
                    if cpt and cpt >= K:
                        hxp.explain('user', history, approaches=[STRATEGY], n=N, imp_type=IMPORTANCE_TYPE, csv_file=utility_csv)
                    # Update history
                    if history.full():
                        # Update history
                        history.get()
                        history.get()
                    history.put(action)

                new_state, reward, done, _, _ = env.step(action)
                env.render()
                cpt += 1

            # Last (B-)HXP
            if COMPUTE_HXP:
                history.put(env.s)
                hxp.explain('user', history, approaches=[STRATEGY], n=N, imp_type=IMPORTANCE_TYPE, csv_file=utility_csv)
