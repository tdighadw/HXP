import argparse
import json
import os

from algorithms import policy_iteration
from env import LabrinthEnv

if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-map', '--map_name', default="corridor", help="Map's dimension (nxn)", type=str, required=False)
    parser.add_argument('-policy', '--policy_name', default="corridor_det_env", help="Common part of policy name", type=str, required=False)
    parser.add_argument('-ep', '--nb_episodes', default=100, help="Maximal number of training episodes", type=int, required=False)
    parser.add_argument('-value_iter', '--value_iteration', default=100, help="Maximal value iteration episodes", type=int, required=False)
    parser.add_argument('-gamma', '--gamma', default=0.8, help="Gamma", type=float, required=False)
    parser.add_argument('-theta', '--theta', default=0.001, help="Theta", type=float, required=False)

    parser.add_argument('-stoch', '--stoch_env', dest="stochastic_env", action="store_true", help="Stochastic environment", required=False)
    parser.add_argument('-no_stoch', '--no_stoch_env', action="store_false", dest="stochastic_env", help="Deterministic environment", required=False)
    parser.set_defaults(stochastic_env=False)

    args = parser.parse_args()

    # Get arguments
    MAP_NAME = args.map_name
    POLICY_NAME = args.policy_name
    NB_EPISODES = args.nb_episodes
    VALUE_ITER = args.value_iteration
    GAMMA = args.gamma
    THETA = args.theta
    STOCH_ENV = args.stochastic_env

    # Path to store policy
    policy_dirpath = "Policies"

    # Init environment
    if STOCH_ENV:
        env = LabrinthEnv(map_name=MAP_NAME, proba=[0.5, 0.5])
    else:
        env = LabrinthEnv(map_name=MAP_NAME)

    # Train
    # Policy iteration
    policy, V = policy_iteration(env, NB_EPISODES, VALUE_ITER, GAMMA, THETA)

    print("Policy : {}".format(policy))
    print("V: {}".format(V))

    # Save policy
    policy_list = list(policy.items())
    with open(policy_dirpath+ os.sep + 'policy_' + POLICY_NAME, 'w') as fp:
        json.dump(policy_list, fp)

    del policy, V
