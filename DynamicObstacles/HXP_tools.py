import random
from copy import deepcopy
from itertools import product

# Pixel types
WALL = [2, 5, 0]
GOAL = [8, 1, 0]
EMPTY = [1, 0, 0]
OBSTACLE = [6, 2, 0]

### HXP functions ###

#  Extract from the environment important positions for updating the agent's state
#  Input: environment (DynamicObstacles)
#  Output: goal/agent position (int list), obstacles positions (int list list), front position (int list)
def get_positions(env):
    # goal position
    grid_size = env.unwrapped.grid.width - 2
    goal_pos = (grid_size, grid_size)

    # agent position
    agent_pos = env.unwrapped.agent_pos
    #print(agent_pos)

    # front position
    front_pos = tuple(env.unwrapped.front_pos)
    #print(front_pos)

    # obstacles positions
    obs_pos = []
    for obs in env.unwrapped.obstacles:
        obs_pos.append(obs.cur_pos)

    return goal_pos, agent_pos, obs_pos, front_pos

#  Get from a state-action couple, the entire/part of the transitions available, i.e. the new states with
#  their probabilities
#  Input: agent's state (int list list list), action (int), environment (DynamicObstacles), importance score method
#  (str), number of exhaustive/deterministic steps (int), additional information (dict), importance type (str)
#  Output: list of transition-probability couples (couple list)
def transition(s, a, env, approx_mode, exh_steps=0, det_tr=0, add_info=None, imp_type=None):
    # get positions associated to s
    goal_pos, agent_pos, obs_pos, front_pos = s[1]
    # early stop: s is terminal ----------------------------------------------------------------------------------------
    # agent on goal position
    if agent_pos == goal_pos:
        return [(1.0, s)]

    # agent will hit an object (update agent pos)
    is_front_of_object = object_front(env, obs_pos, front_pos, a)
    if is_front_of_object:
        s_copy = deepcopy(s)
        s_copy[1][1] = front_pos
        return [(1.0, s_copy)]

    # else, generate all available next states -------------------------------------------------------------------------
    transitions = []
    #print('Before update positions: goal {}, agent {}, obs {}, front {}, action {}'.format(goal_pos, agent_pos, obs_pos, front_pos, a))
    positions, nb_tr = update_positions(s, a, exh_steps, det_tr, approx_mode)
    proba = 1 / nb_tr if approx_mode == 'none' or exh_steps else 1.0 if approx_mode == 'last' else probability_scale(1/nb_tr, det_tr)

    # Create state and associated proba
    for pos in positions:
        #print('new positions: {}'.format(pos))
        view = get_view(pos, env)
        transitions.append((proba, [view, pos]))

    return transitions

#  Equally distribute the remaining probability among all transitions
#  Input: list of transitions probabilities (float list), number of transitions (int)
#  Output: list of transitions probabilities (float list)
def probability_scale(proba, nb_tr):
    sum_pr = proba * nb_tr
    if sum_pr != 1.0:
        delta = 1.0 - sum_pr
        return proba + (delta / nb_tr)

    else:
        return proba

#  Create an agent's state based on the positions of different objects in the grid.
#  Dimension of the produced state: (7, 7, 3).
#  Input: list of object positions (int list list), environment (DynamicObstacles)
#  Output: state (int list list list)
def get_view(positions, env):
    def cond_1(i, agent_pos, goal_pos, direction):
        if direction in [0, 2]:
            #print('full wall horizontal: {} -- gaol_pos[0]: {} -- bool {}'.format(agent_pos[0] + i, goal_pos[0], agent_pos[0] + i > goal_pos[0] or agent_pos[0] + i < 1))
            return agent_pos[0] + i > goal_pos[0] or agent_pos[0] + i < 1

        else:
            #print('full wall vertical: {} -- gaol_pos[1]: {} -- bool {}'.format(agent_pos[1] + i, goal_pos[1], agent_pos[1] + i > goal_pos[1] or agent_pos[1] + i < 1))
            return agent_pos[1] + i > goal_pos[1] or agent_pos[1] + i < 1

    def cond_2(i, agent_pos, goal_pos, direction):
        if direction in [1, 2]:
            idx = i #if direction == 1 else j
            return agent_pos[direction - 1] + idx > goal_pos[direction - 1]

        else:
            if direction == 0:
                return agent_pos[1] - i < 1
            else:
                return agent_pos[0] - i < 1

    goal_pos, agent_pos, obs_pos, front_pos = positions
    direction = get_direction(agent_pos, front_pos)
    view_size = env.unwrapped.agent_view_size
    half_view = int(view_size // 2)
    view = []
    #print("goal {} - agent {} - obs - {} front {} - dir {}".format(goal_pos, agent_pos, obs_pos, front_pos, direction))

    # The representation of image state is reversed in comparison to the environment's implementation
    first_loop_range = [half_view, - half_view - 1, -1] if direction in [2, 3] else [- half_view, half_view + 1]
    #print('first_loop_range: {} -- direction {}'.format(first_loop_range, direction))

    for i in range(*first_loop_range):
        # outside grid --> add wall
        if cond_1(i, agent_pos, goal_pos, direction):
            view.append([WALL for _ in range(view_size)])

        else:
            feature = []
            # add in reverse order
            for j in range(view_size - 1 , -1, -1):
                # outside grid (right)
                if cond_2(j, agent_pos, goal_pos, direction):
                    feature.append(WALL)

                # inside grid
                else:
                    if direction == 1:
                        #print('Inside: {}'.format((agent_pos[0] + j, agent_pos[1] + i)))
                        feature.append(compare_cell(agent_pos, goal_pos, obs_pos, (agent_pos[0] + j, agent_pos[1] + i)))
                        #print(agent_pos[0] + j, agent_pos[1] + i)

                    elif direction == 2:
                        #print('Inside: {}'.format((agent_pos[0] + i, agent_pos[1] + j)))
                        feature.append(compare_cell(agent_pos, goal_pos, obs_pos, (agent_pos[0] + i, agent_pos[1] + j)))
                        #print(agent_pos[0] + i, agent_pos[1] + j)

                    elif direction == 3:
                        feature.append(compare_cell(agent_pos, goal_pos, obs_pos, (agent_pos[0] - j, agent_pos[1] + i)))
                        #print(agent_pos[0] - j, agent_pos[1] + i)

                    elif direction == 0:
                        feature.append(compare_cell(agent_pos, goal_pos, obs_pos, (agent_pos[0] + i, agent_pos[1] - j)))
                        #if direction == 0: print((agent_pos[0] + i, agent_pos[1] - j))

            view.append(feature)

    return view

#  Get the number of available transitions and the list of new positions, result of doing 'action' from 's'
#  Input: agent's state (int list list list), action (int), number of exhaustive/approximate steps (int),
#  approximate mode for utility computation (str)
#  Output: list of new positions (int tuple list list), number of available transitions (int)
def update_positions(s, action, exh_steps, det_tr, approx_mode):
    goal_pos, agent_pos, obs_pos, front_pos = s[1]
    #print('initial obs pos: {}'.format(obs_pos))

    #  Get the total number of available transitions, and available transitions ----------------------------------------
    obs_pos_tmp = []
    for pos in obs_pos:
        # neighbor positions size (3, 3)
        neighbors = get_neighbors(pos)
        # number of reachable cells for the Ball (in the grid and not gaol and not agent pos)
        available_neighbors = [n for n in neighbors if 1 <= n[0] <= goal_pos[0] and 1 <= n[1] <= goal_pos[1] and n not in [agent_pos, goal_pos]]
        #print('available neighbors from obs pos {}: {}'.format(pos, available_neighbors))
        obs_pos_tmp.append(available_neighbors)

    # get all combinations of obstacle positions / remove combinations where multiple obstacles are at the same position
    obs_combinations = []
    for com in product(*[l for l in obs_pos_tmp]):
        com = list(com)
        if not same_pos(com, agent_pos):
            #print('comb removed: {}'.format(com))
            obs_combinations.append(com)

    # number of possible transitions
    nb_tr = len(obs_combinations)
    #print('number of possible transitions: {}'.format(nb_tr))

    # Select or not transitions, update positions ----------------------------------------------------------------------
    n, sample = (nb_tr, False) if approx_mode == 'none' or exh_steps else (1, True) if approx_mode == 'last' else (det_tr, True)
    #('sample: {}'.format(sample))
    positions = get_new_positions(obs_combinations, goal_pos, agent_pos, front_pos, action, n, sample)

    return positions, nb_tr

#  Check whether multiple obstacles and the agent have similar position or not
#  Input: obstacles positions (int list list), agent position (int list)
#  Output: (bool)
def same_pos(obs_pos, agent_pos):
    # A ball cannot be placed in the same position of the agent (before performing agent's action)
    if agent_pos in obs_pos:
        return True

    for pos in obs_pos:
        # A ball cannot be placed in the same position of another ball
        if obs_pos.count(pos) > 1:
            return True

    return False

#  Create set of new positions
#  Input:  obstacle positions (int list list), goal / agent / front position (int list), agent's action (int),
#  number of new positions to generate (int)
#  Output: set of positions (int list list list)
def get_new_positions(obs_combinations, goal_pos, agent_pos, front_pos, action, n, sample):
    positions = []
    direction = get_direction(agent_pos, front_pos)
    # update agent/front position --------------------------------------------------------------------------------------
    if action == 2: # forward
        old_agent_pos = agent_pos
        old_front_pos = front_pos
        agent_pos = front_pos
        front_pos = get_front_pos(agent_pos, direction)
        #print('new agent pos: {} -- new front pos: {}'.format(agent_pos, front_pos))

    elif action == 1: # rotate right
        direction =  (direction + 1) % 4
        front_pos = get_front_pos(agent_pos, direction)
        #print('new direction: {} -- new front pos: {}'.format(direction, front_pos))

    else: # rotate left
        direction -= 1
        if direction < 0:
            direction += 4
        front_pos = get_front_pos(agent_pos, direction)
        #print('new direction: {} -- new front pos: {}'.format(direction, front_pos))

    # get all positions ------------------------------------------------------------------------------------------------
    if not sample:
        for i in range(n):
            # specific case: Ball and agent in the same cell --> agent stays in its old position
            if agent_pos in obs_combinations[i]:
                positions.append([goal_pos, old_agent_pos, obs_combinations[i], old_front_pos])
                #print('final agent - goal positions: {} -- {}'.format(old_agent_pos, old_front_pos))

            else:
                positions.append([goal_pos, agent_pos, obs_combinations[i], front_pos])
                #print('final agent - goal positions: {} -- {}'.format(agent_pos, front_pos))

    # sample positions -------------------------------------------------------------------------------------------------
    else:
        idx = min(n, len(obs_combinations)) # avoid to try to sample more transitions than there are
        #print(idx)

        for _ in range(idx):
            r = random.randint(0, len(obs_combinations) - 1)
            # specific case: Ball and agent in the same cell --> agent stays in its old position
            if agent_pos in obs_combinations[r]:
                positions.append([goal_pos, old_agent_pos, obs_combinations[r], old_front_pos])

            else:
                positions.append([goal_pos, agent_pos, obs_combinations[r], front_pos])
            del obs_combinations[r]


    return positions

#  Determine the type of cell
#  Input: goal position (int list), list of obstacle positions (int list list), position of studied cell (int list)
#  Output: Type of cell (image encoding) (int list)
def compare_cell(agent_pos, goal_pos, obs_pos, coord):
    # match goal / agent
    if goal_pos == agent_pos:
        return EMPTY
    # match goal / coord
    if goal_pos == coord:
        return GOAL

    for obs in obs_pos:
        # match obstacle / coord
        if obs == coord:
            return OBSTACLE

    # empty cell of the grid
    return EMPTY

#  Get the current agent's direction
#  Input: agent / front position (int list)
#  Output: agent's direction (int)
def get_direction(agent_pos, front_pos):
    # same line
    if agent_pos[0] == front_pos[0]:
        if agent_pos[1] < front_pos[1]:
            return 2  # right
        else:
            return 0 # left

    # same column
    if agent_pos[1] == front_pos[1]:
        if agent_pos[0] < front_pos[0]:
            return 1 # down
        else:
            return 3 # up

    print('Error: agent direction')
    return None

#  Get the current position in front of the agent
#  Input: agent position (int list), direction (int)
#  Output: front position (int list)
def get_front_pos(agent_pos, direction):
    if direction == 0: # left
        return agent_pos[0], agent_pos[1] - 1

    elif direction == 1: # down
        return agent_pos[0] + 1, agent_pos[1]

    elif direction == 2: # right
        return  agent_pos[0], agent_pos[1] + 1

    else: # up
        return agent_pos[0] - 1, agent_pos[1]

#  Check whether a state is terminal or not
#  Input: state (int list list list), environment (DynamicObstacles), additional information (dict)
#  Output: (bool)
def terminal(s, env, add_info):
    goal_pos, agent_pos, obs_pos, front_pos = s[1]
    # agent is on goal position / is outside the grid / hits an obstacle
    return agent_pos == goal_pos or is_outside_grid(agent_pos, env) or agent_pos in obs_pos

#  Check whether there is an object in front of the agent (it's terminal when the action is 'move forward')
#  Input: environment (DynamicObstacles), obstacles positions (int list list), front position (int list), action (int)
#  Output: (bool)
def object_front(env, obs_pos, front_pos, action):
    if action == 2:
        is_wall = is_outside_grid(front_pos, env)
        return front_pos in obs_pos or is_wall
    else:
        return False

#  Check whether the position is outside the grid or not
#  Input: position (int list), environment (DynamicObstacles)
#  Output: (bool)
def is_outside_grid(pos, env):
    grid_size = env.unwrapped.grid.width - 2
    return not (0 < pos[0] < grid_size + 1 and 0 < pos[1] < grid_size + 1)

#  Compute the argmax of an array
#  Input: an array (list)
#  Output: index of the maximum value (int)
def argmax(array):
    array = list(array)
    return array.index(max(array))

#  Multiple-tasks function to update some data during the HXP process
#  Input: environment (DynamicObstacles), agent, location of the modification in the HXP process (str),
#  state-action list (int list list list), additional information (dict)
#  Output: variable
def preprocess(env, agent, location, s_a_list=None, add_info=None):
    #  Flatten the 2D list into 1D list
    if location == 'pre_locally_minimal_paxp':
        add_info['paxp_cur_state_pos'] = s_a_list[0]
        return to_flatten_state(s_a_list[0][0])

    #  From 1D list to 2D list
    if location == 'post_locally_minimal_paxp':
        return to_state(s_a_list[0], env.unwrapped.agent_view_size)[0]

    if location == 'weak_paxp':
        return env, agent, s_a_list

    return env, agent

#  From state to flattened state
#  Input: state (int list list list)
#  Output: flattened state (int list list)
def to_flatten_state(state):
    flat_state = []
    for elm in state:
        flat_state.extend(elm)
    return flat_state

#  From flattened state to state
#  Input: flattened state (int list list), view/grid size (int), starting column/line index (int), mid-line index (int)
#  Output: state and partial state only composed of the grid (int list list list)
def to_state(flat_state, view_size, grid_size=0, start_column_idx=0, start_line_idx=0, mid_line=0):
    state = []
    grid_state = []
    new_flat_state = []
    feature = []

    # build an entire flattened state
    if grid_size:
        line_idx = [i for i in range(start_line_idx, min(mid_line + grid_size, view_size))]
        column_range = [start_column_idx, view_size]
        grid_line = []
        j = 0
        #print('line_idx:', line_idx)
        #print('column range:', column_range)
        #print('mid line:', column_range)

        for i in range(view_size):
            # add a line of outside grid
            if i not in line_idx:
                #print('i not in line_range:', i)
                for _ in range(view_size):
                    new_flat_state.append(WALL)

            else:
                #print('i in line_range:', i)
                for _ in range(start_column_idx):
                    new_flat_state.append(WALL)
                for _ in range(*column_range):
                    new_flat_state.append(flat_state[j])
                    grid_line.append(flat_state[j])
                    j += 1

                # agent's position
                if i == mid_line:
                    #print("grid line before pop: ", grid_line)
                    new_flat_state.pop()
                    grid_line.pop()
                    new_flat_state.append(EMPTY)
                    grid_line.append(EMPTY)
                    #print('grid line after', grid_line)
                    j -= 1

                # create the potential grid
                if len(grid_line) == grid_size:
                    grid_state.append(grid_line)
                    grid_line = []
    #print('new_flat_state', new_flat_state)

    # build the state
    f_state = flat_state if not grid_size else new_flat_state
    for idx, elm in enumerate(f_state):
        feature.append(elm)
        if len(feature) == view_size:
            state.append(feature)
            feature = []

    return state, grid_state

#  Check whether an importance score can be computed or not
#  Input: action (int), importance type (str), additional information (dictionary)
#  Output: (bool)
def constraint(action, imp_type, add_info):
    return False

#  Get available actions from a state
#  Available actions are similar no matter the state
#  Input: state (int list list list), environment (DynamicObstacles)
#  Output: action list (int list)
def get_actions(s, env):
    return [i for i in range(env.action_space.n)]

#  Render the most important action(s) / transition(s)
#  Input: state action list to display (int list list list), environment (DynamicObstacles), agent,
#  importance type (str), runtime (float), additional information (dict)
#  Output: None
def render(hxp, env, agent, imp_type, runtime, add_info):
    # Render
    for s_a_list, i in hxp:
        print("Timestep {}".format(i))
        hxp_render(s_a_list[0][0])
        print("    ({})".format(["Turn Left", "Turn Right", "Move Forward"][s_a_list[1]]))
        if imp_type == 'transition':
            hxp_render(s_a_list[2][0])
    # Runtime
    print("-------------------------------------------")
    print("Explanation achieved in: {} second(s)".format(runtime))
    print("-------------------------------------------")
    return

### Backward HXP functions ###

#  Get all (or sub-part) of the states whose match with the set features of v
#  Input: environment (DynamicObstacles), studied state (int list list list), index of value to remove (int), number of
#  states to sample (int), additional information (dict)
#  Output: list of states (int list list list list)
def sample(env, v, i, n, add_info):
    tmp_value = v[i]
    v[i] = None
    allow_terminal = True

    # get useful information for the limitation of states
    agent_view_size = env.unwrapped.agent_view_size
    grid_size = env.unwrapped.grid.width - 2
    mid_line = int(agent_view_size // 2) # line of the agent's position
    n_obstacles = env.unwrapped.n_obstacles

    # get the flattened coordinates of potential grid cells
    line_range = [max(0, mid_line - grid_size + 1), min(mid_line + grid_size, agent_view_size)]
    column_range = [agent_view_size - min(grid_size, agent_view_size), agent_view_size]
    #print('line_range: {}'.format(line_range))
    #print('column_range: {}'.format(column_range))

    potential_grid_features_indexes = [a * agent_view_size + b for a in range(*line_range) for b in range (*column_range) if not (a == mid_line and b == agent_view_size - 1)]
    potential_grid_features = [v[c] for c in potential_grid_features_indexes]
    # print('potential_grid_features: {}'.format(potential_grid_features))
    # print('potential_grid_features_indexes: {}'.format(potential_grid_features_indexes))

    # in case a sample is similar to actual state, set the same positions
    cur_state = add_info['paxp_cur_state_pos']

    # constant feature
    if i not in potential_grid_features_indexes:
        print('i not in potential grid features: {}'.format(i))
        v[i] = tmp_value
        return []

    else:
        # grid / ball / wall / goal
        feature_values = [EMPTY, OBSTACLE, WALL, GOAL]
        samples = []

        # reduce the number of combinations by considering grid features only
        none_features_indexes = [k for k in range(len(v)) if v[k] is None and k in potential_grid_features_indexes]
        none_features_nb = len(none_features_indexes)

        # remove useless feature values --------------------------------------------------------------------------------

        # only one goal in the grid / only n obstacles in the grid
        obs_cpt, goal_cpt = count_obs_goal(potential_grid_features)
        if goal_cpt:
            feature_values.remove(GOAL)
        if obs_cpt == n_obstacles:
            feature_values.remove(OBSTACLE)
        del obs_cpt, goal_cpt

        # compute the set of all possible combinations of values for None features -------------------------------------

        for values in product(feature_values, repeat=none_features_nb):
            #print('values: {}'.format(values))
            is_valid, p_state = valid(values, potential_grid_features, potential_grid_features_indexes, none_features_indexes, n_obstacles, agent_view_size, grid_size, cur_state)

            if is_valid and (allow_terminal or not terminal(p_state, env, None)):
                samples.append(p_state)
                # early stop: only n samples are required
                if len(samples) == n:
                    break

        v[i] = tmp_value
        return samples

#  Create a point and check whether a point is valid.
#  Input: features values to test (int list), partial state features/indexes (int list list), none features indexes
#  (int list), number of obstacles (int), view/grid size (int), baseline state (int list list list)
#  Output: (bool), agent's state and set of positions (list)
def valid(values, potential_grid_feats, potential_grid_feats_indexes, none_features_indexes, n_obstacles, view_size, grid_size, cur_state):
    #  Check validity of visual state ----------------------------------------------------------------------------------
    invalid = False, [[]]
    # early stop: more obstacles than n_obstacles
    val_obs_cpt, val_goal_cpt = count_obs_goal(values)
    pgf_obs_cpt, pgf_goal_cpt = count_obs_goal(potential_grid_feats)
    if val_obs_cpt + pgf_obs_cpt > n_obstacles or val_goal_cpt + pgf_goal_cpt > 1:
        #print('more obstacles than n_obstacles')
        return invalid

    del val_obs_cpt, val_goal_cpt, pgf_obs_cpt, pgf_goal_cpt

    # create point p
    p = deepcopy(potential_grid_feats)
    i = 0
    for idx, elm in enumerate(p):
        if elm is None:
            if potential_grid_feats_indexes[idx] in none_features_indexes:
                p[idx] = values[i]
                i += 1
            #  constant feature: agent position
            else:
                p[idx] = EMPTY
    del i

    # early stop: p similar to starting state
    mid_line = int(view_size // 2)
    start_column_idx = view_size - min(grid_size, view_size)
    start_line_idx = max(0, mid_line - grid_size + 1)
    p_state, grid_state = to_state(p, view_size, grid_size, start_column_idx, start_line_idx, mid_line) # TODO ok
    if p_state == cur_state[0]:
        #print('similar to starting state')
        return True, cur_state

    # start: first line / column index which is inside the potential grid features
    # first: first line / column index which is inside the grid
    first_line_idx = -1
    first_col_idx = -1
    indexes_grid_lines = []
    tmp_idx = 0
    rel_obs_pos = [] # used for the computation of positions
    rel_goal_pos = []

    for idx, line in enumerate(grid_state):
        #print('line {} : {}'.format(idx, line))
        line_idx = start_line_idx + idx
        indexes_grid_columns = []
        y_goal = []

        for i, elm in enumerate(line):
            if elm != WALL:
                indexes_grid_columns.append(start_column_idx + i)
            if elm == OBSTACLE:
                rel_obs_pos.append((line_idx, start_column_idx + i))
            if elm == GOAL:
                y_goal.append(start_column_idx + i)

        # inside grid
        if indexes_grid_columns:
            #print('indexes_grid_columns: {}'.format(indexes_grid_columns))
            indexes_grid_lines.append(line_idx)
            # first line
            if first_line_idx < 0:
                first_line_idx = line_idx
                first_col_idx = indexes_grid_columns[0]

            # other lines
            else:
                # early stop: non-square shape, mismatch between columns
                if indexes_grid_columns[0] != first_col_idx:
                    #print('non square grid shape')
                    #hxp_render(p_state)
                    return invalid

            # early stop: a wall inside the grid
            if indexes_grid_columns != [i for i in range(indexes_grid_columns[0], view_size)]:
                #print('a wall is inside the grid')
                #hxp_render(p_state)
                return invalid

            # save goal position
            if y_goal:
                rel_goal_pos = [line_idx, y_goal[0]]
                #print('Visible goal (y_goal) --> coords: {}'.format(rel_goal_pos))

            tmp_idx = indexes_grid_columns

    indexes_grid_columns = tmp_idx # keep the number of columns

    # early stop: line of wall split the grid in two parts
    if indexes_grid_lines != [i for i in range(indexes_grid_lines[0], indexes_grid_lines[0] + len(indexes_grid_lines))]:
        #print('Line of walls split the grid')
        #hxp_render(p_state)
        return invalid

    # early stop: invalid number of grid lines
    if len(indexes_grid_lines) not in [i for i in range(min(mid_line + 1, grid_size), min(view_size, grid_size) + 1)]:
        #('invalid number of grid lines: idx grid lines {} --'.format(indexes_grid_lines))
        #hxp_render(p_state)
        return invalid

    # early stop: the whole grid is visible and too many items are missing
    if len(indexes_grid_lines) == grid_size and len(indexes_grid_columns) == grid_size:
        p_obs_cpt, p_goal_cpt = count_obs_goal(p)
        if (p_goal_cpt != 1 or p_obs_cpt != n_obstacles):
            #print('Missing objects on the grid')
            #hxp_render(p_state)
            return invalid
        del p_obs_cpt, p_goal_cpt

    # if there is a visible goal, it must be in a corner of the grid
    if rel_goal_pos and len(indexes_grid_lines) == min(grid_size, view_size):
        x_rel_goal_positions = [indexes_grid_lines[0], indexes_grid_lines[-1]]
        y_rel_goal_positions = [indexes_grid_columns[0]]
        if len(indexes_grid_columns) == min(grid_size, view_size): y_rel_goal_positions.append(indexes_grid_columns[-1])

        # early stop: goal not in a corner
        if rel_goal_pos[0] not in x_rel_goal_positions or rel_goal_pos[1] not in y_rel_goal_positions:
            #print('goal not in the corner of the grid')
            #hxp_render(p_state)
            return invalid

    #  Sample only one set of valid positions from the visual state ----------------------------------------------------

    # goal pos
    positions = [(grid_size, grid_size)] # goal position doesn't change
    # relative agent pos
    limit_line_dist, limit_column_dist = min(mid_line - indexes_grid_lines[0], indexes_grid_lines[-1] - mid_line), min(len(indexes_grid_columns) - 1, grid_size - len(indexes_grid_columns))
    # specific case: grid view > view size (several distances are possible)
    if grid_size > view_size and limit_line_dist == mid_line:
        potential_limit_line_dist = [mid_line + i for i in range(grid_size - int(view_size // 2) + 1)]
        limit_line_dist = random.choice(potential_limit_line_dist)

    # Choose an agent direction and absolute position
    directions = [i for i in range(4)]
    agent_positions = []
    while not agent_positions and directions:
        direction = random.choice(directions) # sample a direction
        #print('tested direction: {}'.format(direction))

        agent_positions = get_potential_positions(p_state, grid_size, view_size, limit_line_dist, limit_column_dist, direction, indexes_grid_columns, indexes_grid_lines, rel_goal_pos)
        directions.remove(direction)
        #print('new directions: {}'.format(directions))
        #print()

    #print('final list of potential positions: {}'.format(agent_positions))
    agent_pos = random.choice(agent_positions)
    #print('chosen one: {}'.format(agent_pos))
    positions.append(agent_pos)

    # absolute obs pos
    obs_pos = []
    agent_rel_pos = (mid_line, len(p_state[0]) - 1)
    for pos in rel_obs_pos:
        abs_pos = get_abs_obs_pos(pos, agent_rel_pos, agent_pos, direction)
        obs_pos.append(abs_pos)
    #print('in view obs pos {}:'.format(obs_pos))

    # specific case: obstacles are not visible by the agent.
    # generate a sample of position(s) for the 'hidden' obstacle(s)
    if len(obs_pos) != n_obstacles:
        hidden_grid_pos = get_hidden_grid_pos(agent_pos, direction, view_size, grid_size)

        for i in range(len(obs_pos), n_obstacles):
            abs_pos = random.choice(hidden_grid_pos) # random provides some errors of feature evaluation during PAXp computation
            obs_pos.append(abs_pos)
            hidden_grid_pos.remove(abs_pos)

    #print('out of view obs pos: {}'.format(obs_pos))

    positions.append(obs_pos)
    #print('final obs pos: {}'.format(obs_pos))

    # front pos
    front_pos = get_front_pos(agent_pos, direction)
    positions.append(front_pos)

    #print('front pos: {}'.format(front_pos))
    #print('positions: {}'.format(positions))

    # add positions to the agent's state
    p_state = [p_state, positions]

    return True, p_state

#  Count the number of obstacles and goals
#  Input: features list (int list list list)
#  Output: number of obstacles, goals (int)
def count_obs_goal(l):
    tmp_obs_cpt = 0
    tmp_goal_cpt = 0
    for elm in l:
        if elm == GOAL:
            tmp_goal_cpt += 1

        elif elm == OBSTACLE:
            tmp_obs_cpt += 1

    return tmp_obs_cpt, tmp_goal_cpt

#  Get the absolute position of a visible obstacle from the agent's state
#  Input: relative coordinates of the agent and obstacle (int list), agent absolute coordinates (int list),
#  agent's direction (int)
#  Output: absolute coordinates of the obstacle (int couple)
def get_abs_obs_pos(rel_obs_pos, rel_agent_pos, abs_agent_pos, direction):

    # get relative distance between pos
    if direction in [1, 3]: # down, up
        c, l = rel_agent_pos[0] - rel_obs_pos[0], rel_agent_pos[1] - rel_obs_pos[1]
    else: # left, right
        l, c = rel_agent_pos[0] - rel_obs_pos[0], rel_agent_pos[1] - rel_obs_pos[1]

    # get absolute obs pos
    # x coord
    if direction in [0, 3]: # left, up
        abs_x_obs = abs_agent_pos[0] - l
    else: # down, right
        abs_x_obs = abs_agent_pos[0] + l

    # y coord
    if direction in [2, 3]: # right, up
        abs_y_obs = abs_agent_pos[1] + c
    else: # left, down
        abs_y_obs = abs_agent_pos[1] - c

    #print('Abs goal pos: {}-{}'.format(abs_x_obs, abs_y_obs))
    return abs_x_obs, abs_y_obs

#  Get absolute positions of the hidden cells of the grid (A cell is hidden if it doesn't appear in the agent's state)
#  Input: agent's position (int list), agent's direction (int), view/grid size (int)
#  Output: hidden cells positions (int list list)
def get_hidden_grid_pos(agent_pos, direction, view_size, grid_size):
    visible_positions = []
    nb_col, nb_r_line, nb_l_line = get_view_grid_dimensions(agent_pos[0], agent_pos[1], view_size, grid_size, direction)
    #print('view grid dimensions: col {} - r_line {} - l_line {}'.format(nb_col, nb_r_line, nb_l_line))

    # loop on columns
    for c in range(nb_col):
        #print('column: {}'.format(c))
        idx = agent_pos[1] if direction in [1,3] else agent_pos[0]
        loop_range =  [idx - nb_r_line, idx + nb_l_line + 1] if direction in [0, 1] else [idx - nb_l_line, idx + nb_r_line + 1]

        for l in range(*loop_range):
            #print('line : {}'.format(l))
            if direction in [1,3]:
                if direction == 1:
                    visible_positions.append((agent_pos[0] + c, l))
                else:
                    visible_positions.append((agent_pos[0] - c, l))
            else:
                if direction == 2:
                    visible_positions.append((l, agent_pos[1] + c))
                else:
                    visible_positions.append((l, agent_pos[1] - c))

    # remove agent position from visible positions (specific case: obstacle in the agent position)
    visible_positions.remove(agent_pos)
    #print('visible positions: {}'.format(visible_positions))

    # avoid goal position and visible positions except the agent position (terminal state)
    hidden_positions = [(i, j) for i in range(1, grid_size + 1) for j in range(1, grid_size + 1) if (i, j) not in visible_positions and (i, j) != (grid_size, grid_size)]
    #print('hidden positions: {}'.format(hidden_positions))

    return hidden_positions

#  Get the set of potential absolute agent position
#  Input: state (int list list list), grid/view size (int), minimal agent's distance to the grid limit (int), agent's
#  direction (int), indexes of visible lines/columns (int list), goal visible to the agent (bool)
#  Output: set of potential agent positions (int list list)
def get_potential_positions(s, grid_size, view_size, line_dist, column_dist, direction, idx_col, idx_li, visible_goal):

    def goal_pos_cond(positions, i, j, mid_line, view_size, grid_size, direction):
        return not any([get_abs_obs_pos(pos, [mid_line, view_size - 1], [i, j],
                                                    direction) == (grid_size, grid_size) for pos in positions])
    positions = []
    mid_line = int(view_size // 2)
    for i in range(1, grid_size + 1):
        for j in range(1, grid_size + 1):

            l_dist = min(i - 1, grid_size - i) # minimal dist to a border
            c_dist = min(j - 1, grid_size - j) # minimal dist to a border

            if (l_dist == column_dist and c_dist == line_dist) or (l_dist == line_dist and c_dist == column_dist):
                #print('potential position: {}'.format((i,j)))

                # check whether dimensions map to the state (according to the agent position and direction)
                nb_col, nb_r_line, nb_l_line = get_view_grid_dimensions(i, j, view_size, grid_size, direction)
                #print('grid dimensions: nb_col {}, nb_r_line {}, nb_l_line {}'.format(nb_col, nb_r_line, nb_l_line))

                idx_mid = idx_li.index(mid_line)
                #print('idx_mid {} from idx_li {}'.format(idx_mid, idx_li))
                r, l = idx_li[:idx_mid], idx_li[(idx_mid + 1):]

                if len(idx_col) == nb_col and len(r) == nb_r_line and len(l) == nb_l_line:
                    # new position is found
                    if not visible_goal:
                        # specific case: agent on the goal position (only one column is visible and the agent is in a corner of the grid)
                        if i == j == grid_size and nb_col == 1 and (not nb_r_line or not nb_l_line):
                            #print('on the goal: r_line {}, l_line {}, nb_col {}'.format(nb_r_line, nb_l_line, nb_col))
                            positions.append((i, j))

                        # absolute position is not in the visible part of the grid
                        else:
                            col_pos = [view_size - 1, view_size - nb_col] if nb_col != 1 else [view_size - 1]
                            potential_rel_goal_pos = [[a,b] for a in [mid_line - nb_r_line, mid_line + nb_l_line] for b in col_pos if [a,b] != [i,j]]

                            # visible obstacle not in the same position as goal
                            rel_obs_pos = [[i,j] for i in range(len(s)) for j in range(len(s[0])) if s[i][j] == OBSTACLE]
                            if (goal_pos_cond(potential_rel_goal_pos, i, j, mid_line, view_size, grid_size, direction) and
                                    goal_pos_cond(rel_obs_pos, i, j, mid_line, view_size, grid_size, direction)):
                                #print('not obstacles in goal position and no (3,3) pos')
                                positions.append((i, j))

                    # check whether the position is valid regarding the visible goal position
                    else:
                        # goal located at (3,3)
                        if get_abs_obs_pos(visible_goal, [mid_line, view_size - 1], [i, j], direction) == (grid_size, grid_size):
                            #print('valid position: {} - {}'.format(i, j))
                            positions.append((i, j))

    return positions

#  Get the dimensions of the view of the grid in an agent's state according to its position and direction TODO:compress
#  Input: x,y coordinates (int), agent's view size (int), grid size (int), agent's direction (int)
#  Output: number of visible columns (int), relative positions to walls (int)
def get_view_grid_dimensions(x, y, view_size, grid_size, direction):
    nb_col, nb_r_line, nb_l_line = -1, -1, -1
    half_view = int(view_size // 2)

    if direction == 0: # left
        nb_col = min(view_size, y)
        nb_r_line = min(half_view, x - 1)
        nb_l_line = min(half_view, grid_size - x)

    elif direction == 1: # down
        nb_col = min(view_size, grid_size - (x - 1))
        nb_r_line = min(half_view, y - 1)
        nb_l_line = min(half_view, grid_size - y)

    elif direction == 2: # right
        nb_col = min(view_size, grid_size - (y - 1))
        nb_r_line = min(half_view, grid_size - x)
        nb_l_line = min(half_view, x - 1)

    elif direction == 3: # up
        nb_col = min(view_size, x)
        nb_r_line = min(half_view, grid_size - y)
        nb_l_line = min(half_view, y - 1)

    return  nb_col, nb_r_line, nb_l_line

### Predicates ###

#  Check whether the agent reach the goal position or not
#  Input: state (int list list list), additional information (dict)
#  Output: (bool)
def success(s, info):
    return s[1][0] == s[1][1]

#  Check whether the agent has a collision or not with an obstacle
#  Input: state (int list list list), additional information (dict)
#  Output: (bool)
def collision(s, info):
    env = info['env']
    # agent is outside the grid (wall collision) or on an obstacle position (Ball collision)
    return is_outside_grid(s[1][1], env) or s[1][1] in s[1][2]

#  Check whether Balls are close to the agent. Here 'close' means that the Balls is directly available from one
#  'move forward' action. The Ball must be visible from the agent
#  Input: state (int list list list), additional information (dict)
#  Output: (bool)
def close_balls(s, info):
    # get the type of 3 cells adjacent to the agent
    adjacent_cells_type = get_cell_type(s[0], 'agent adjacent')
    #print(adjacent_cells_type)
    return adjacent_cells_type.count(6) > 0

#  Check whether the agent is in a defined specific position or not
#  Input: state (int list list list), additional information (dict)
#  Output: (bool)
def specific_position(s, info):
    spec_pos = info['pred_params'][0]
    #print('spec_pos: ', spec_pos, 'agent_pos: ', s[1][1])
    return spec_pos == s[1][1]

#  Check whether the agent is not in a defined specific position or not
#  Input: state (int list list list), additional information (dict)
#  Output: (bool)
def avoid_specific_position(s, info):
    spec_pos = info['pred_params'][0]
    #print('spec_pos: ', spec_pos, 'agent_pos: ', s[1][1])
    return spec_pos != s[1][1]

#  Check whether a predicate defined via PAXp holds or not
#  Input: state (int list list list), additional information (dict)
#  Output: (bool)
def redefined_predicate(s, info):
    predicate = info['redefined_predicate']
    #print()
    #print('predicate : {}'.format(predicate))
    #print('state: {}'.format(s))

    for i in range(len(predicate)):
        for j in range(len(predicate[0])):

            feature = predicate[i][j]
            if feature is not None and feature != s[0][i][j]:
                return False

    return True

#  Get the type of several cells (image encoding)
#  Input: state (int list list list), location of the cell to describe (str)
#  Output: several cell types (int list or int list list)
def get_cell_type(s, location='front'):
    # mid-line
    mid = int(len(s) // 2)
    #print(s)

    if location == 'agent adjacent':
        #print('mid, ', mid)
        return [s[mid][-2][0], s[mid + 1][-1][0], s[mid - 1][-1][0]]

    else:
        return s[0][mid][-2][0]

#  Get neighbor positions of a specific position. The result is the positions of the square of size 3x3
#  centered on pos (pos is omitted).
#  Input: position (int list)
#  Output: neighbor positions (int list list)
def get_neighbors(pos):
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            neighbors.append((pos[0] + i, pos[1] + j))

    del neighbors[4]
    #print('all neighbors: {}'.format(neighbors))

    return neighbors

### Find histories for a specific predicate ###

#  Verify if the last state from a proposed history respects a predicate
#  Input: state (int list list list), predicate (str), additional information (dict)
#  Output: (bool)
def valid_history(s, predicate, info):
    if predicate == 'success':
        return success(s, info)
    elif predicate == 'collision':
        return collision(s, info)
    elif predicate == 'specific_position':
        return specific_position(s, info)
    elif predicate == 'avoid_specific_position':
        return avoid_specific_position(s, info)
    elif predicate == 'close_balls':
        return close_balls(s, info)

### Additional functions for state rendering ###

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

#  Highlight the background of a text.
#  Input: text which need a colored background (str), a color (str) and use or not of different colors (bool)
#  Output: Text with colored background (str)
def colorize(text, color_, small=True):
    if small:
        num = color2num[color_]
        return (f"\x1b[{num}m{text}\x1b[0m")
    else:
        num = color_
        return (f"\x1b[48;5;{num}m{text}\x1b[0m")

#  Rendering agent's state in the console output
#  Input: state (int list list list)
#  Output: None
def hxp_render(s):
    print()
    #  Print objects
    for i in range(len(s)):
        str_tmp = '|'
        for j in range(len(s[0])):
            if s[i][j] is None: # backward hxp debug
                str_tmp += colorize(' ', "yellow")

            else:
                object_type = s[i][j][0]
                if object_type != 1:
                    # wall
                    if object_type == 2:
                        color = "white"
                    # ball
                    elif object_type == 6:
                        color = "blue"
                    # goal
                    elif object_type == 8:
                        color = "green"
                    str_tmp += colorize(' ', color)

                else:
                    # agent pos
                    if i == int(len(s) // 2) and j == len(s[0]) - 1:
                        str_tmp += colorize(' ', "red")
                    # empty
                    else:
                        str_tmp += ' '

            str_tmp += '|'
        print(str_tmp)

    print()
    return
