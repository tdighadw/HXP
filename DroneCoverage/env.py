
from gymnasium import Env
from gymnasium.spaces import Discrete
import numpy as np
import random

#  Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STOP = 4

#  Items of the environment
OUT_OF_GRID = 0
EMPTY = 1
DRONE = 2
TREE = 3

#  Action space
Actions = [LEFT, DOWN, RIGHT, UP, STOP]

#  Set of maps
MAPS = {
    "10x10": ["S---T---TS",
              "-T--------",
              "----------",
              "-----T----",
              "----------",
              "--T-------",
              "--------TT",
              "----------",
              "------T---",
              "S--------S"],

    "30x30": ["S-------TTT---S--T------TT---S",
              "--------------------S---------",
              "-TT--S-----------------------S",
              "--------TT------S---TTT-------",
              "--------------------T----S----",
              "S-----------T-----------------",
              "------------T-----------------",
              "---TT---STT------T---S--TT----",
              "-TT---------------------------",
              "S------------T--S------------S",
              "---------S-----------TT-------",
              "--------TT--------------------",
              "--S-----TT---S---------S-----T",
              "---------T------------T------T",
              "------------------------------",
              "S---T----S-------T------TT---S",
              "-------------------S----------",
              "---S---------T-------------T--",
              "------T-------------TTT-------",
              "-----TTTT----S----T------S----",
              "S------T----------T-----------",
              "------------------------------",
              "----TT--S--------T---S--TT----",
              "--------------S---------------",
              "S---T-------------------------",
              "----T--------TTT-S-----------S",
              "------S------TT----------TT---",
              "------------TTT---------------",
              "------------------------------",
              "S-----T-------S---------TT---S"]

}

# Following 2 functions are inspired by
# - https://stackabuse.com/how-to-print-colored-text-in-python/
# - https://github.com/openai/gym/blob/master/gym/utils/colorize.py

#  Dict of colors
color2num = dict(
    gray=40,
    red=41,
    green=42,
    yellow=43,
    blue=44,
    magenta=45,
    cyan=46,
    white=47,
)

#  Highlight the background of text. It's used to display the cover of a drone
#  Input: text which need a colored background (str), a color (str) and use or not of different colors (Boolean)
#  Output: Text with colored background (str)
def colorize(text, color_, small=True):
    if small:
        num = color2num[color_]
        return (f"\x1b[{num}m{text}\x1b[0m")
    else:
        num = color_
        return (f"\x1b[48;5;{num}m{text}\x1b[0m")

#  Color a text in yellow. Function only used for 30x30 map. It highlights drones with a perfect cover.
#  Input: text to highlight (str)
#  Output: highlighted text (str)
def yellow(text, color_=None):
    if color_ is None:
        return (f"\x1b[1;33m{text}\x1b[0m")
    else:
        return text
    
class DroneAreaCoverage(Env):

    def __init__(self, map_name="10x10", windless=False, wind_probas=[0.3, 0.2, 0.4, 0.1]):
        #  Only updated when render function is called
        self.render_map = np.asarray(MAPS[map_name], dtype="c")
        self.nRow, self.nCol = len(self.render_map), len(self.render_map[0])
        #  Action space
        self.action_space, self.wind_action_space = Discrete(len(Actions)), Discrete(len(Actions) - 1)
        #  Initialize map
        self.init_map()
        #  List of last actions
        self.last_actions = None
        #  Transition probability
        self.windless = windless

        if self.windless:
            self.P = None
        else:
            self.P = wind_probas

    #  Initialize the map (int list list) based on the render map (the map does not contain drones for now)
    #  Input: None
    #  Output: None
    def init_map(self):
        map = []
        for i in range(len(self.render_map)):
            line = []
            for j in range(len(self.render_map[0])):

                if bytes(self.render_map[i, j]) == b"T":
                    line.append(TREE)  # Tree
                else:
                    line.append(EMPTY)  # Otherwise

            map.append(line)

        self.map = map
        return

    #  Initialize observations of Agents
    #  Input: agents (Agent list)
    #  Output: None
    def initObs(self, agents):
        for agent in agents:
            #  Agent: only set an observation
            agent.set_obs([agent.view(agent.position), agent.position])
        return

    #  Initialize position of Agents
    #  Input: agents (Agent list), put agents in pre-defined position or not (bool)
    #  Output: None
    def initPos(self, agents, rand):
        for agent in agents:
            #  Set position
            if rand:
                agent.position = self.get_random_position()

            else:
                agent.position = self.get_starting_position()

            # Map's update
            self.map[agent.position[0]][agent.position[1]] = DRONE
        return

    #  Set a specific position to an agent
    #  Input: agent (Agent) and specific position (int list)
    #  Output: None
    def set_initPos(self, agent, position, collision=False):
        agent.position = position
        # Map's update

        #if collision: print('map pos before update:', self.map[position[0]][position[1]], 'collision:', collision)
        if self.map[position[0]][position[1]] != TREE:
            #print('put drone in position ' + str(position))
            self.map[agent.position[0]][agent.position[1]] = DRONE

        # empty the position
        if collision and self.map[position[0]][position[1]] != TREE:
            #print('empty position due to collision')
            self.map[agent.position[0]][agent.position[1]] = EMPTY

        return

    #  Update coordinates of a drone
    #  Input: coordinates and an action (int)
    #  Output: new coordinates (int list)
    def inc(self, row, col, a):
        if a == LEFT:
            col = max(col - 1, 0)

        elif a == DOWN:
            row = min(row + 1, self.nRow - 1)

        elif a == RIGHT:
            col = min(col + 1, self.nCol - 1)

        elif a == UP:
            row = max(row - 1, 0)

        return [row, col]

    #  Reset environment and agents' observation
    #  Input: agents (Agent list), put agents in pre-defined position or not (boolean)
    #  Output: None
    def reset(self, agents, rand):
        # Initialize the map
        self.init_map()
        positions = []

        #  Select a position for each agent
        for agent in agents:
            if not rand:  # Start at a predefined position "S"
                position = self.get_starting_position()

            else:  # Start in any position except trees and drones positions
                position = self.get_random_position()

            #  Reset dead, position, map position value
            agent.dead = False
            agent.position = position
            self.map[position[0]][position[1]] = DRONE
            positions.append(position)

        #  Set new states for each agent
        for i in range(len(positions)):
            agents[i].set_obs([agents[i].view(positions[i]), positions[i]])

        return

    #  Perform action for all agents, update the map and their observations and check collisions
    #  Input: agents (Agent list) and their actions (int list), device (str), type of transition function (str)
    #  Output: the old, temporary (before the wind moves the agent) and new states (int list list), list
    #  used to know if it's the end of an episode (boolean list) and a copy of the current map (int list list)
    def step(self, agents, actions, device="cpu", move="all"):
        #  Initialization
        states = []
        temp_states = []
        new_states = []
        new_positions = []
        map_copy = None
        #  Get agents states
        for agent in agents:
            states.append(agent.get_obs())

        #  Update map and positions
        for i in range(len(actions)):
            self.map[states[i][1][0]][states[i][1][1]] = EMPTY
            new_position = self.inc(states[i][1][0], states[i][1][1], actions[i])
            new_positions.append(new_position)

        #  Due to transition probabilities, new positions change
        if not self.windless:
            for i in range(len(new_positions)):
                #  Transition
                transition = np.random.choice(4, p=self.P)
                #  Change position if actions are not opposite and not 'stop' or if action is 'stop'
                if move == "all":
                    if actions[i] == STOP or not (actions[i] - 2 == transition or transition - 2 == actions[i]):
                        new_positions[i] = self.inc(new_positions[i][0], new_positions[i][1], transition)

                else:
                    if actions[i] != STOP and not(actions[i] - 2 == transition or transition - 2 == actions[i]):
                        new_positions[i] = self.inc(new_positions[i][0], new_positions[i][1], transition)

        #  Update self.map, agents.dead and dones
        dones = self.collisions(agents, new_positions)
        #  Update observations of agents
        for i in range(len(agents)):
            agents[i].position = new_positions[i]
            agents[i].set_obs([agents[i].view(agents[i].position), agents[i].position])
            new_states.append(agents[i].get_obs())

        #  Update for rendering the environment
        self.last_actions = actions

        return states, temp_states, new_states, dones, map_copy

    #  Check collisions and update map values and dead attributes of agents
    #  Input : agents (Agent list), agents positions (int list list)
    #  Output : list used to know if it's the end of an episode (bool list)
    def collisions(self, agents, positions):

        unique_positions = []
        dones = []
        i = 0
        for position in positions:
            if agents[i].dead:
                dones.append(True)

            else:
                #  Fill unique_positions
                if position not in unique_positions:
                    unique_positions.append(position)

                #  Check collision with a tree
                if self.map[position[0]][position[1]] == TREE:
                    dones.append(True)
                    agents[i].dead = True

                else:
                    dones.append(False)
                    self.map[position[0]][position[1]] = DRONE

            i += 1

        #  Check collision between 2 drones
        if len(unique_positions) < len(positions):
            #  Extract coordinates of collision
            for i in range(len(positions)):
                if positions.count(positions[i]) > 1:
                    dones[i] = True
                    agents[i].dead = True
                    self.map[positions[i][0]][positions[i][1]] = EMPTY

        return dones

    #  List all agents which have an imperfect cover
    #  Input: agents (Agent list)
    #  Output: agents with an imperfect cover (Agent list)
    def agentsImperfectCover(self, agents):
        imprfct_agents = []
        for agent in agents:
            view = agent.get_obs()[0]

            #  Get only the wave range matrix
            index_range = (agent.view_range - agent.wave_range) // 2
            sub_view = [s[index_range:-index_range] for s in view[index_range:-index_range]]

            #  Another drone in range or a tree in coverage area zone
            if sum([sub_list.count(DRONE) for sub_list in view]) > 1 or sum([sub_list.count(TREE) for sub_list in sub_view]) > 0:
                imprfct_agents.append(agent)

        return imprfct_agents

    #  Display to the user the current state of the environment. There are two different ways for rendering the
    #  environment. It depends on the number of agents on the map.
    #  Input: agents (Agent list)
    #  Output: None
    def render(self, agents):
        small_agents_nbr = len(agents) <= 8
        imprfct_agents = []

        # Determine number of imperfect agents (if there is more than 8 agents)
        if not small_agents_nbr:
            imprfct_agents = self.agentsImperfectCover(agents)

        if small_agents_nbr or len(imprfct_agents) <= 8:
            colors = ["blue", "green", "red", "yellow", "cyan", "magenta", "gray", "black"]

        else:
            colors = [str(i) for i in range(1, 256)]

        #  Print current actions of each drone
        if self.last_actions is not None:
            str_actions = ["Left", "Down", "Right", "Up", "Stop"]
            string = " "
            cpt = 0
            for i in range(len(agents)):
                if imprfct_agents and agents[i] in imprfct_agents or not imprfct_agents:
                    #  Display action (or Dead)
                    if agents[i].dead:
                        string += colorize("Dead", colors[i])

                    else:
                        string += colorize(str_actions[self.last_actions[i]], colors[i])

                    string += " "
                    #  Avoid a long line of actions
                    if cpt % 9 == 0 and cpt not in [len(self.last_actions) - 1, 0]:
                        string += "\n\n "

                    cpt += 1

            print(string)
        print()

        #  Update render_map to the current map
        render_map = self.render_map.tolist()
        render_map = [[c.decode("utf-8") for c in line] for line in render_map]

        for i in range(len(agents)):
            if not agents[i].dead:
                position = agents[i].get_obs()[1]
                render_map[position[0]][position[1]] = "D"
                #  Colorize covered cells
                if imprfct_agents and agents[i] not in imprfct_agents:
                    render_map[position[0]][position[1]] = yellow(render_map[position[0]][position[1]])

                else:
                    self.color_coverageArea(render_map, position, agents[i], colors[i%len(colors)], small=small_agents_nbr)

        #  Render
        for line in range(self.nRow):
            row_str = ""
            for column in range(self.nCol):
                row_str = row_str + render_map[line][column]
            print(row_str)

        print()
        return

    #  Color the coverage area of the agent
    #  Input: render map (np array), position of the agent (int list), the agent (Agent), a color (str)
    #  and a different choice of color (bool)
    #  Output: None
    def color_coverageArea(self, map, position, agent, color, small):
        #  Color each cell under constraints
        wave_range_index = agent.convert_index(agent.wave_range//2)

        for i in wave_range_index:
            if 0 <= position[0] + i < len(self.map):  # Out of bounds condition

                for j in wave_range_index:
                    if 0 <= position[1] + j < len(self.map):   # Out of bounds condition
                        # Cell is neither occupied by a tree nor a drone
                        if map[position[0] + i][position[1] + j] != "T" and map[position[0] + i][position[1] + j] != "D":
                            map[position[0] + i][position[1] + j] = colorize(map[position[0] + i][position[1] + j], color, small=small)
        return

    #  Compute the maximum reachable cumulative reward
    #  Input: agents (Agent list), list used to know if it's the end of an episode (bool list) and
    #  reward type (str)
    #  Output: reward (int)
    def max_env_reward(self, agents, dones=None, reward_type="B"):
        max_reward = 0
        for i in range(len(agents)):
            if dones is None:
                if reward_type == "A":
                    max_reward += 1

                else:
                    max_reward += len(agents) - 1

            else:
                if not dones[i]:
                    if reward_type == "A":
                        max_reward += 1

                    else:
                        max_reward += len(agents) - 1

        return max_reward

    #  At a timestep, compute the reward for each agent
    #  Input: agents (Agent list), actions (int list), list used to know if it's the end of an episode (bool list)
    #  and reward's type (str)
    #  Output: list of reward (float list)
    def getReward(self, agents, actions, dones, reward_type="B"):
        rewards = []
        i = 0
        for agent in agents:
            #  Crash of an agent
            if dones[i]:
                if reward_type == "A":
                    rewards.append(-1)

                else:
                    rewards.append(-(len(agents)-1))

            else:
                #  Initialization
                reward = 0
                max_cells_highlighted = (agent.wave_range * agent.wave_range - 1)
                max_agents_inrange = agent.view_range**2 - 1
                max_agents_inrange = min(max_agents_inrange, len(agents)-1)
                view = agent.get_obs()[0]

                #  Get only the wave range matrix for computing reward (3x3 matrix)
                index_range = (agent.view_range - agent.wave_range) // 2
                sub_view = [s[index_range:-index_range] for s in view[index_range:-index_range]]
                #  Count highlighted cells in 3x3 matrix
                cells_highlighted = sum(sub_list.count(EMPTY) for sub_list in sub_view)

                if reward_type == "A":
                    #  Penalty if at least one different drone is in view range
                    if sum([sub_list.count(DRONE) for sub_list in view]) > 1:
                        reward = -1

                    #  Penalty Stop action without perfect cover
                    if self.windless:
                        if actions[i] == STOP and cells_highlighted != max_cells_highlighted:
                            reward = -1

                    #  Cover reward
                    if reward == 0:
                        #  Perfect cover
                        if max_cells_highlighted == cells_highlighted:
                            reward = 1

                else:
                    #  Penalty for each drone(s) in view range
                    reward -= (sum([sub_list.count(DRONE) for sub_list in view]) - 1) / max_agents_inrange * (len(agents)-1)

                    #  Cover reward
                    if max_cells_highlighted == cells_highlighted:
                        #  Perfect cover
                        reward += len(agents)-1

                    else:
                        reward += (cells_highlighted / max_cells_highlighted) * (len(agents)-2)

                rewards.append(reward)

            i += 1

        return rewards

    #  Set a list of actions (this function is used for the render method)
    #  Input : actions (int list)
    #  Output : None
    def set_lastactions(self, actions):
        self.last_actions = actions
        return

    #  Get the action space
    #  Input : None
    #  Output : action space (Discrete)
    def get_actionspace(self):
        return self.action_space

    #  Get the list of predefined available starting free positions "S" and randomly select one
    #  Input: None
    #  Output: a randomly-chosen position (int list)
    def get_starting_position(self):
        positions = []
        for i in range(len(self.render_map)):
            for j in range(len(self.render_map[0])):

                if bytes(self.render_map[i, j]) == b"S" and self.map[i][j] != DRONE:  # 2: drone is already here
                    positions.append([i, j])

        return random.choice(positions)

    #  Get a random position where a drone can be in the map
    #  Input: None
    #  Output: a position (int list)
    def get_random_position(self):
        i, j = random.randint(0, len(self.map)-1), random.randint(0, len(self.map)-1)

        while bytes(self.render_map[i, j]) == b"T" or self.map[i][j] == DRONE:  # 2: drone already here | T: tree
            i, j = random.randint(0, len(self.map)-1), random.randint(0, len(self.map)-1)

        return [i, j]

    #  Clear the map (remove each drone)
    #  Input: None
    #  Output: None
    def clear_map(self):
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] == DRONE:
                    self.map[i][j] = EMPTY
        return
