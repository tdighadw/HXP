import argparse
import os
from env import MyFrozenLake
from agent import Agent

if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-map', '--map_name', default="4x4", help="Map's dimension (nxn)", type=str, required=False)
    parser.add_argument('-policy', '--policy_name', default="4x4map_test", help="Common part of policy name", type=str, required=False)
    parser.add_argument('-ep', '--nb_episodes', default=10000, help="Number of training episodes", type=int, required=False)

    parser.add_argument('-equiprobable', '--equiprobable', dest="equiprobable", action="store_true", help="Equiprobable transitions", required=False)
    parser.add_argument('-no_equiprobable', '--no_equiprobable', action="store_false", dest="equiprobable", help="Equiprobable transitions", required=False)
    parser.set_defaults(equiprobable=False)

    parser.add_argument('-features', '--many_features', dest="many_features", action="store_true", help="Several features to define agent's state", required=False)
    parser.add_argument('-no_features', '--no_many_features', action="store_false", dest="many_features", help="Only one feature to define the agent's state", required=False)
    parser.set_defaults(many_features=False)

    args = parser.parse_args()
    
    # Get arguments
    MAP_NAME = args.map_name
    POLICY_NAME = args.policy_name
    NB_EPISODES = args.nb_episodes
    EQUIPROBABLE = args.equiprobable
    FEATURES = args.many_features

    # Paths to store Q table
    agent_Q_dirpath = "Q-tables" + os.sep + "Agent"

    # Env initialization
    if EQUIPROBABLE:
        env = MyFrozenLake(map_name=MAP_NAME, many_features=FEATURES)
    else:
        env = MyFrozenLake(map_name=MAP_NAME, slip_probas=[0.2, 0.6, 0.2], many_features=FEATURES)

    # Agent initialization
    agent = Agent(POLICY_NAME, env)

    # Train
    agent.train(NB_EPISODES)

    # Save Q table
    agent.save(agent_Q_dirpath)

    # Delete agent
    del agent
