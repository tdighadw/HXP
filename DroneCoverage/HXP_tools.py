import random
from copy import deepcopy
from itertools import product

OUT_OF_GRID = 0
EMPTY = 1
DRONE = 2
TREE = 3

# Predicate parameters:
#  - agent_id_pov : display its most important action / transition (value between 1 and len(agents))
#  - agent_id_pred : observe its respect for the predicate (value between 0 and len(agent))
#  When agent_id_pred is set to 0, we study the global version of the predicate
#  When agent_id_pov is different from agent_id_pred, we study the impact of agent_id_pov
#  in the agent_id_pred respect of the predicate (except for 'region' predicate)
#  For the 'region' predicate, agent_id_pred indicates the region to study

### Generate the length-k scenarios ###

#  Get from a state-action couple, the entire/part of the transitions available, i.e. the new states associated with
#  their probabilities
#  Input: agent's state (int list list list), action (int), environment (DroneCoverage), importance score method (str),
#  number of exhaustive/deterministic steps (int), additional information (dictionary), importance type (str)
#  Output: list of transition-probability couples (couple list)
def transition(s, a, env, approx_mode, exh_steps=0, det_tr=0, add_info=None, imp_type=None):
    # format: [transitions for 3 agents, transitions for 4 agents]
    all_transitions = add_info['all_transitions']
    agents = add_info['agent']
    agent_id_pov = add_info['pred_params'][0]
    history_transitions = add_info['history_transitions']
    states = get_states(agents)
    new_positions = []
    transitions = []
    info = {}
    #print('initial positions: {}'.format([s[1] for s in states]))
    #print('actions: {}'.format(a))

    #  Update map and positions
    for i in range(len(a)):
        if not agents[i].dead:
            env.map[states[i][1][0]][states[i][1][1]] = 1
            new_position = env.inc(states[i][1][0], states[i][1][1], a[i])
            new_positions.append(new_position)

        # agent is dead, position unchanged
        else:
            new_positions.append(states[i][1])

    #print('intermediate positions: {}'.format(new_positions))

    # Look all possible transitions from s
    if approx_mode == 'none' or exh_steps:
        # Other drones perform the most probable transition (total of 4 transitions)
        if not all_transitions:
            info['others_tr'] = history_transitions
            info['other_tr'] = argmax(env.P)
            info['agent_id'] = agent_id_pov

            for idx, p in enumerate(env.P):
                agents_copy, new_positions_copy, env_copy = copies(agents, new_positions, env)
                info['agent_tr'] = idx
                transitions = update_transitions(env_copy, agents_copy, a, transitions, new_positions_copy, p, info)
                #print(transitions)

        # All drones perform all transitions (total of 256 transitions)
        else:
            for t in all_transitions[1]:
                #print('check!')
                agents_copy, new_positions_copy, env_copy = copies(agents, new_positions, env)
                p = prod([t[i] for i in range (len(new_positions_copy))])
                info['others_tr'] = [env.P.index(t[i]) for i in range (len(new_positions_copy))]
                transitions = update_transitions(env_copy, agents_copy, a, transitions, new_positions_copy, p, info)

        return transitions

    else:
        # Look for the most probable transition
        if approx_mode == 'last':
            return extract_transitions(1, all_transitions, a, env, agents, new_positions, approx_mode)

        # Select the 'det' most probable transition(s)
        else:
            return extract_transitions(det_tr, all_transitions, a, env, agents, new_positions, approx_mode, agent_id_pov)

#  Check whether a state is terminal or not
#  Input: state (int list list list), environment (DroneCoverage), additional information (dict)
#  Output: (bool)
def terminal(s, env, add_info):
    agent_id_pov = add_info['pred_params'][0]
    # Check state of one drone
    s_agent = s[agent_id_pov - 1]

    return s_agent[0][len(s_agent[0]) // 2][len(s_agent[0][0]) // 2] != DRONE

#  Compute the argmax of an array
#  Input: an array (list)
#  Output: index of the maximum value (int)
def argmax(array):
    array = list(array)
    return array.index(max(array))

#  Extract n most probable transitions
#  Input: number of transition to extract (int), list of exhaustive transitions (int list list), action (int),
#  environment (DroneCoverage), agents (Agent list), positions (int list list), approximate mode (str), agent id (int)
#  Output: most probable transition(s) (int list list)
def extract_transitions(n, all_transitions, a, env, agents, positions, approx_mode, agent_id=None):
    info = {}
    most_probable = []
    agent_id_p, p, tmp_idx = -1, -1, -1
    info['agent_id'] = agent_id
    info['other_tr'] = argmax(env.P)
    transitions = [[idx,p] for idx, p in enumerate(env.P)] if not all_transitions else deepcopy(all_transitions)
    #print(transitions)

    while n != len(most_probable):
        # ------ Extract max and argmax of transition -----------------

        # Other drones perform the most probable transition in any case
        if not all_transitions:
            # Most probable for agent_id
            p_transitions = [p for _,p in transitions]
            agent_id_p, tmp_idx = max(p_transitions), argmax(p_transitions)
            info['agent_tr'] = transitions[tmp_idx][0]
            tmp_cpt = p_transitions.count(agent_id_p)

        # All transitions are possible
        else:
            tmp_p_list = [prod([p for p in l]) for l in transitions[1]]
            tmp_idx, p = argmax(tmp_p_list), max(tmp_p_list)
            #print(transitions[1][tmp_idx])
            info['others_tr'] = [env.P.index(transitions[1][tmp_idx][i]) for i in range(len(agents))]
            tmp_cpt = tmp_p_list.count(p)

        proba = agent_id_p if not all_transitions else p

        # Only one transition is the current most probable one
        if tmp_cpt == 1:
            agents_copy, new_positions_copy, env_copy = copies(agents, positions, env)
            most_probable = update_transitions(env_copy, agents_copy, a, most_probable, new_positions_copy,
                                               proba, info)
            # remove the transition
            if not all_transitions:
                transitions.remove([info['agent_tr'], proba])
                #print(transitions)

            else:
                #print(transitions[1][tmp_idx])
                del transitions[1][tmp_idx]
        else:
            # There are more transitions than wanted (random pick)
            if tmp_cpt > n - len(most_probable):
                # ------ Extract max transition -----------------
                if not all_transitions:
                    info['agent_tr'] = random.choice([idx for idx, t in enumerate(transitions) if t == agent_id_p])

                else:
                    tmp_idx = random.choice([idx for idx, pr in enumerate(tmp_p_list) if pr == p])
                    info['others_tr'] = [env.P.index(transitions[tmp_idx][i]) for i in range(len(agents))]

                agents_copy, new_positions_copy, env_copy = copies(agents, positions, env)
                most_probable = update_transitions(env_copy, agents_copy, a, most_probable, new_positions_copy, proba, info)

                # remove the transition
                if not all_transitions:
                    transitions.remove(transitions[info['agent_tr']])

                else:
                    transitions.remove(transitions[1][tmp_idx])

            # Add all transitions in most_probable
            else:
                tmp_list = []
                for idx, p in enumerate(transitions):
                    condition = p == proba if not all_transitions else prod(p) == proba

                    if condition:
                        tmp_list.append(idx)
                        agents_copy, new_positions_copy, env_copy = copies(agents, positions, env)
                        if not all_transitions:
                            info['agent_tr'] = idx

                        else:
                            info['others_tr'] = [env.P.index(p[i]) for i in range(len(agents))]
                        most_probable = update_transitions(env_copy, agents_copy, a, most_probable, new_positions_copy, proba, info)

                for idx in tmp_list:
                    transitions.remove(transitions[idx])

    # Probability distribution
    sum_pr = sum([p for p, s in most_probable])
    if sum_pr != 1.0:
        delta = 1.0 - sum_pr
        add_p = delta / len(most_probable)

        for elm in most_probable:
            elm[0] += add_p

    return most_probable

#  Get agents' states
#  Input: agents (Agent list)
#  Output: list of states (int list list list list)
def get_states(agents):
    states = []
    #  Get agents states
    for agent in agents:
        states.append(agent.get_obs())

    return states

#  Perform deep copies of agents, positions and environment
#  Input: agents (Agent list), positions (int list list), environment (DroneCoverage)
#  Output: agents (Agent list), positions (int list list), environment (DroneCoverage)
def copies(agents, positions, env):
    return deepcopy(agents), deepcopy(positions), deepcopy(env)

#  Compute the product of probabilities
#  Input: probabilities list (float list)
#  Output: (float)
def prod(l):
    p = 1
    for elm in l:
        p *= elm
    return p

#  Get transition for a specific agent
#  Input: agent id (int), additional information (dict)
#  Output: transition (int)
def select_transition(idx, info):
    # Get infos
    agent_id = 0 if 'agent_id' not in info else info['agent_id']
    others_tr = [] if 'others_tr' not in info else info['others_tr']
    agent_tr = -1 if 'agent_tr' not in info else info['agent_tr']
    other_tr = -1 if 'other_tr' not in info else info['other_tr']
    #print('agent_id {} -- agent tr {}'.format(agent_id,agent_tr))

    # Transition for the agent to explain
    if agent_id and agent_id == idx + 1:
        return agent_tr

    # Transition for other drones
    else:
        if others_tr:
            return others_tr[idx]
        else:
            return other_tr

#  Update the list of transitions
#  Input: environment (DroneCoverage), agents (Agent list), actions (int list), current transitions list
#  (int list list), agents positions (int list list), joint transitions probability (float), additional information (dict)
#  Output: transitions list (int list list)
def update_transitions(env, agents, actions, transitions, positions, proba, info=None):
    # info contains: agent_id / history_transitions / agent_tr / others_tr
    new_states = []
    #print('agent tr: {}'.format(info['agent_tr']))

    for i in range(len(positions)):
        transition = select_transition(i, info)
        #print('transition {} for agent {}'.format(transition, i+1))
        #  Change position if actions are not opposite and not 'stop' or if action is 'stop'
        if actions[i] != 4 and not (actions[i] - 2 == transition or transition - 2 ==  actions[i]):
            positions[i] = env.inc(positions[i][0], positions[i][1], transition)

    #  Update self.map, agents.dead and dones
    env.collisions(agents, positions)

    #  Update observations of agents
    for i in range(len(agents)):
        new_states.append(
            [agents[i].view(positions[i], optional_map=env.map), positions[i]])

    #print(new_states)
    transitions.append([proba, new_states])

    return transitions

### Compute importance score ###

#  Check whether an importance score can be computed or not
#  Input: action (int), importance type (str), additional information (dictionary)
#  Output: (bool)
def constraint(action, imp_type, add_info):
    agent_id_pov = add_info['pred_params'][0]
    return action[agent_id_pov - 1] == 4 and imp_type == 'transition'

#  Sample n valid states
#  Input: environment (DroneCoverage), partial state (int list list list), index of feature to remove from v (int),
#  number of samples to generate (int), additional information (dict)
#  Output: list of states (int list list list list)
def sample(env, v, i, n, add_info=None):
    tmp_value = v[i]
    v[i] = None
    v_copy = deepcopy(v)

    # Extract information
    agents = add_info['agent']
    view_range = agents[0].view_range
    mid_view_range = int(view_range) // 2
    coords = [v[-2], v[-1]]
    view = v[:-2]

    # agent has a collision with another agent
    if v[view_range * mid_view_range + mid_view_range] == EMPTY:
        max_drone_nb = len(agents) - 2

    # agent is not dead / had a collision with a TREE / feature is None
    else:
        max_drone_nb = len(agents)

    #print('coord: {} / view: {}'.format(coords, view))
    none_coord_cpt = coords.count(None)
    none_view_cpt = view.count(None)
    is_set_pos = not none_coord_cpt
    is_set_view = not none_view_cpt

    # Do not evaluate constant features:
    # the agent's position is set and the feature is whether a tree nor 'out grid'
    if is_set_pos and tmp_value in [TREE, OUT_OF_GRID] and i not in [len(v) - 1, len(v) - 2]:
        #print('constant feature is not evaluated: {}'.format(tmp_value))
        v[i] = tmp_value
        return []

    else:
        view_feature_values = [OUT_OF_GRID, EMPTY, DRONE, TREE]

        # Remove useless features & build none dict --------------------------------------------------------------------

        #print()
        #print('########### STEP: remove useless features  and build none_dict ##########')

        # Map features cannot change with an already set position
        if is_set_pos:
            #print('fixed position: remove TREE / OUT OF GRID')
            view_feature_values.remove(TREE)
            view_feature_values.remove(OUT_OF_GRID)
            # set to whether TREE or OUT_OF_GRID the None features which can't change
            #print()
            #print('Set constant features')

            for j in range(view_range):
                # get line from the environment
                env_line = get_line(env, coords[0] - mid_view_range + j, coords[1], view_range)
                #print('env_line ',  coords[0] - mid_view_range + i, ': ', env_line)

                for k, f in enumerate(env_line):
                    #print('i * view_range + j', i * view_range + j)
                    if f == TREE and v_copy[j * view_range + k] is None:
                        v_copy[j * view_range + k] = TREE

                    elif f == OUT_OF_GRID and v_copy[j * view_range + k] is None:
                        v_copy[j * view_range + k] = OUT_OF_GRID

            #print('new v_copy: ', v_copy)

        # max number of TREE in a line
        max_tree_nb = max([l.count(TREE) for l in env.map])
        #print('max number of tree in a line in the env: {}'.format(max_tree_nb))

        # line: [feature_vals, line, drone_coord, nb_none]
        none_dict = {line: [deepcopy(view_feature_values), [], [], 0] for line in range(view_range)}
        is_central_view_feature_None = False
        line = 0
        tree_line_cpt = 0
        drones_cpt = 0
        first_line_feature, last_line_feature = -1, -1

        for idx in range(len(v_copy) - 2):
            j = idx % view_range if idx else -1 # avoid considering the first feature as a line
            feature = v_copy[idx]
            # Keep the first feature value of the line:
            if j in [0, -1]:
                first_line_feature = v_copy[idx]

            # Generate line
            none_dict[line][1].append(feature)
            # Increment number of None features
            if feature is None:
                none_dict[line][-1] += 1
                # central agent is None: this line must have the priority to change nb_max_drone
                if idx == view_range * mid_view_range + mid_view_range:
                    none_dict[line][-1] += view_range  # It will always be the first line to consider
                    is_central_view_feature_None = True

            # Detected drone
            elif feature == DRONE:
                drones_cpt += 1
                if line != j or line != mid_view_range:
                    none_dict[line][2].append([line, j])

            # Detected tree
            elif feature == TREE:
                tree_line_cpt += 1

            # Change line and remove useless feature values in case of None feature membership to the line
            if j == view_range - 1:
                last_line_feature = v_copy[idx]
                #print('full line: {}'.format(none_dict[line][1]))

                if not is_set_pos:
                    # Maximal number of tree is already reached in this line
                    #print('nb trees in line {}: {}'.format(line, tree_line_cpt))
                    if tree_line_cpt == max_tree_nb:
                        #print('There are already {} TREE in the line -> remove TREE value'.format(max_tree_nb))
                        none_dict[line][0].remove(TREE)

                    # None features are only in the middle of the grid
                    if first_line_feature is not None and last_line_feature is not None and none_dict[line][-1]:
                        #print('None features are only in the middle of the grid -> remove OUTOFGRID value')
                        none_dict[line][0].remove(OUT_OF_GRID)

                # Reset variables
                tree_line_cpt = 0
                first_line_feature, last_line_feature = -1, -1

                line += 1

        del line, tree_line_cpt, first_line_feature, last_line_feature, j
        # No more drones are allowed
        if drones_cpt == max_drone_nb:
            #print('DRONE value removed (already {} drones in fixed features)!'.format(max_drone_nb))
            #print('View: {}'.format(view))
            for line in range(view_range):
                none_dict[line][0].remove(DRONE)

        # Loop on combinations -----------------------------------------------------------------------------------------

        #print('########### STEP: loop on combinations ##########')

        # Sort by decreasing order of the number of features and remove lines without any None feature
        sorted_indexes = sorted([key for key, val in none_dict.items() if val[-1]], key=lambda x: none_dict[x][-1],
                                reverse=True)
        #print(sorted_indexes)

        if is_central_view_feature_None:
            none_dict[sorted_indexes[0]][-1] -= view_range

        # Generate (part) of the states
        samples = []
            # get information
        items_limit = [max_tree_nb, max_drone_nb]
        sub_state_fixed  = [is_set_view, is_set_pos]
        if is_set_pos:
            coords_info = [-1, coords, -1]

        else:
            set_axis, set_value = (0, coords[0]) if coords[0] is not None else (1, coords[1]) if coords[1] is not None else (-1, None)
            coords_info = [set_axis, set_value, [j for j in range(env.nRow)]]

        #print('coords_info: ', coords_info)
        #print('start rec dc product')
        rec_dc_product(None,[], 0, samples, drones_cpt, sorted_indexes, none_dict, n, items_limit, sub_state_fixed, view_range, env, coords_info, add_info['pred_params'][0]) # print_pos
        #print('nb positions tested: {} - set of valid positions {}'.format(len(print_pos),set(print_pos)))
        #print('valid positions in these samples: {}'.format(set(print_pos)))
        #print('value {} removed {}'.format(i, tmp_value))
        #print('end of rec dc product')

        v[i] = tmp_value

        del none_dict, coords_info, items_limit, sub_state_fixed, view_range, sorted_indexes, drones_cpt

        return samples

#
#  items_limit = max_tree_nb, max_drone_nb
#  sub_state_fixed = is_set_view, is_set_pos
#  coords info = [x/y, value, feature_values]
#  Produce each combination of none features values (for view and position) and store the valid states
#  (recursive function)
#  Input: current list of product of none features values of position(s)/lines (int list list), index of the current line
#  studied (int), list of valid states (int list list list list), current number of visible drones (int),
#  lines indexes (int list), lines dictionary (line dict), maximal number of samples (int), maximal number of
#  trees/drones (int list), already set view/position (bool list), agent's view range (int),
#  environment (DroneCoverage), position info (int list), agent id (int)
#  Output: stop function (bool)
def rec_dc_product(position, lines_prod, i, result, drones_cpt, li_indexes, li_dict, n, items_limit, sub_state_fixed, view_range, env, coords_info, agent_id): #print_pos
    # Stop case
    if 0 < n == len(result):
        return True

    # Position found or fixed
    if position is not None or sub_state_fixed[1]:
        #print('position of our agent found {} - {}'.format(position, sub_state_fixed[1]))

        # View found or fixed
        if i >= len(li_indexes) or sub_state_fixed[0]:
            #print('state valid --> begin state extraction')
            #print('test combination: {} - {} - max_drone {}'.format(lines_prod, [li_dict[elm][1] for elm in li_dict], items_limit[1]))
            extract_states(result, position, li_dict, lines_prod, li_indexes, n, agent_id, items_limit[1], env)
            #print('end of state extraction')

            return True

        # Generate a view (line by line)
        else:
            pos = position if position is not None else tuple(coords_info[1])
            for prod in dc_view_product(li_dict[li_indexes[i]][0], drone_cpt=drones_cpt, items_limit=items_limit, repeat=li_dict[li_indexes[i]][-1]):
                # conditions: no wrong 'out of grid' positions, match line with topography, maximal number of drones not exceeded
                #print('nb drones: {}'.format(drones_cpt))
                #print(li_dict, li_indexes, i)
                #print(li_indexes)
                #print(i)
                #print('drone limit {}'.format(items_limit[-1]))
                valid, drones_line_cpt, new_drone_limit = valid_line(prod, pos, li_dict[li_indexes[i]][1], li_indexes[i], drones_cpt, items_limit, view_range, env)
                #print('nb drones: {}'.format(drones_cpt))
                #print('new drone limit {}'.format(new_drone_limit))
                #print('tested prod: {}'.format(prod))

                if valid:
                    lines_prod.append(prod)
                    rec_dc_product(pos, lines_prod, i + 1, result, drones_cpt + drones_line_cpt, li_indexes, li_dict, n, [items_limit[0], new_drone_limit], sub_state_fixed, view_range, env, coords_info, agent_id) #print_pos

                    if 0 < n == len(result): return True
                    del lines_prod[-1]

        return True

    # Generate a position
    else:
        #print('Generate a position')
        r = 2 if coords_info[0] == -1 else 1
        #print('Test positions')

        for pos in dc_coord_product(coords_info[-1], env=env, fixed_value=coords_info[:2], repeat=r):
            # test the topography of the lines in the view with the proposed coords
            #print('Test position: {}'.format(pos))
            valid = valid_coords([li_dict[li][1] for li in li_dict], pos, view_range, env)
            #print('i: {} - lines_prod: {} empty'.format(i, lines_prod))

            if valid:
                #print('valid!: {}'.format(pos))
                rec_dc_product(pos, lines_prod, i, result, drones_cpt, li_indexes, li_dict, n, items_limit, sub_state_fixed, view_range, env, coords_info, agent_id)

        return True

#  Generate a state and get all the drones positions. Update the list of valid states
#  Input: list of valid states (int list list list list), agent position (int list), line dictionary (line dict), none
#  features values per line (int list list), line indexes (int list), maximal number of states to generate
#  (useless parameter) (int), agent id (int), maximal number of drones (int), environment (DroneCoverage)
#  Output: None
def extract_states(result, agent_pos, line_dict, line_prod, line_indexes, n, agent_id, max_drone_nb, env):
    line_dict_copy = deepcopy(line_dict)
    drones_collision = False

    # Create point p and get relative visible agents positions ---------------------------------------------------------
    p_state = [None, agent_pos]
    view = []
    relative_agent_positions = []

    for i, (_, line, drone_coord, nb_none) in enumerate(line_dict_copy.values()):
        # add relative agent position(s)
        if drone_coord: relative_agent_positions.extend(drone_coord)

        # already set line
        if not nb_none:
            view.append(line)

        # build the line
        else:
            tmp_line = line
            idx = line_indexes.index(i)
            k = 0

            for j, feature in enumerate(tmp_line):
                if feature is None:
                    tmp_line[j] = line_prod[idx][k]
                    # add relative agent position(s) which is not the agent position
                    if tmp_line[j] == DRONE and (i != j or j != int(len(tmp_line) // 2)):
                        #print('None drone detected: line {} - idx {}'.format(tmp_line, j))
                        relative_agent_positions.append([i, j])
                    k += 1

            view.append(tmp_line)

    p_state[0] = view
    mid_range = int(len(view[0]) // 2)
    dead_drone_coord = []
    #print('p_state: {}'.format(p_state))

    # specific case: collision of the central drone with another drone
    if view[mid_range][mid_range] == EMPTY:
        #print('collision with our agent')
        relative_agent_positions.append([mid_range, mid_range])
        drones_collision = True
        dead_drone_coord.append(list(agent_pos))

    # specific case: collision of the central agent with a TREE
    if view[mid_range][mid_range] == TREE:
        #print('tree')
        dead_drone_coord.append(list(agent_pos))

    # Get agents positions ---------------------------------------------------------------------------------------------

    # Visible positions
    agent_positions = get_absolute_drone_positions(relative_agent_positions, agent_pos, mid_range)

    max_drone = max_drone_nb + 1 if drones_collision else max_drone_nb
    # Invisible positions
    invisible_drone_nb = max_drone - len(agent_positions)
    #print('max_drone in view range', max_drone)

    if len(agent_positions) == 4:
        print('agent_positions: ', agent_positions)
        print('invisible_drone_nb: {}'.format(invisible_drone_nb))

    # several combination of positions exist / sample positions
    if invisible_drone_nb > 0:
        #print('drone(s) are invisible --> generate potential positions')
        # absolute positions in view range (different from TREE and in environment map)
        view_positions = [[agent_pos[0] + i, agent_pos[1] + j] for i in range(-mid_range, mid_range+1) for j in range(-mid_range, mid_range+1) if 0 <= agent_pos[0] + i < env.nRow and 0 <= agent_pos[1] + j < env.nCol]

        # 1 map configuration per state
        for i in range(1):
            dead_drone_coord_copy = deepcopy(dead_drone_coord)
            combination = deepcopy(agent_positions)
            cpt_positions = deepcopy(invisible_drone_nb)

            while cpt_positions:
                x, y = random.randint(0, env.nRow - 1), random.randint(0, env.nCol - 1)
                while [x, y] in view_positions and cpt_positions < 2:
                    x, y = random.randint(0, env.nRow - 1), random.randint(0, env.nCol - 1)

                # two drones crash in view range
                if [x, y] in view_positions:
                    #print('in-view crash {}'.format([x, y]))
                    combination.append([x, y])
                    cpt_positions -= 1

                # deal with collision
                if env.map[x][y] == TREE or [x, y] in combination:
                    dead_drone_coord_copy.append([x, y])

                #print('dead drone:', dead_drone_coord)
                combination.append([x, y])
                cpt_positions -= 1

            # pov agent position
            combination.insert(agent_id - 1, list(agent_pos))
            #print('dead drone: ', dead_drone_coord_copy)
            #print('final combination: ', combination)
            #print('combination: ', combination, 'dead_drone_coord: ', dead_drone_coord)
            result.append([combination, dead_drone_coord_copy])

            # stop the position generation when the amount of samples is reached
            if len(result) == n:
                break

        #print('end of generation of potential positions')

    # all drones are in view range / only one set of positions exists
    else:
        #print('All drones in view range!')
        #print()

        # pov agent positions
        agent_positions.insert(agent_id - 1, list(agent_pos))

        # deal with dead attribute
        for ag_pos in agent_positions:
            if env.map[ag_pos[0]][ag_pos[1]] == TREE and ag_pos not in dead_drone_coord:
                dead_drone_coord.append([ag_pos[0], ag_pos[1]])

            elif agent_positions.count(ag_pos) > 1 and ag_pos not in dead_drone_coord:
                dead_drone_coord.append(ag_pos)

        #print('agent positions: ', agent_positions)
        #print('dead drone: ', dead_drone_coord)
        result.append([agent_positions, dead_drone_coord])

    #print('result len: ', len(result))
    #print()
    del line_dict_copy
    return

#  Based on the drone's view and its position, get the absolute positions of the other visible drones
#  Input: relative agents positions (int list list), agent's position (int list), mid-view range (int)
#  Output: absolute agents positions (int list list)
def get_absolute_drone_positions(relative_agent_positions, agent_pos, mid_range):
    #print('relative positions found: {}'.format(relative_agent_positions))
    abs_positions = []
    for rel_pos in relative_agent_positions:
        abs_positions.append([agent_pos[0] - mid_range + rel_pos[0], agent_pos[1] - mid_range + rel_pos[1]])

    #print('absolute positions found: {}'.format(abs_positions))
    return abs_positions

#  Check whether the generate coordinate is a valid one or not
#  Input: agent's view (int list list), position (int list), agent's view range (int), environment (DroneCoverage)
#  Output: (bool)
def valid_coords(view, position, view_range, env):
    #print('test match between fixed view and proposed position: {}'.format(position))
    mid_range = int(view_range // 2)
    #print('position tested: ', position)
    #print(view)

    # Fixed view match with coordinates (TREE and OUT_OF_GRID must be at similar positions)
    for i in range(view_range):

        # get line from the environment
        env_line = get_line(env, position[0] - mid_range + i, position[1], view_range)
        #print('len(env_line) =', len(env_line))
        #print('env_line from position {},{}: {}'.format(position[0] - mid_range + i, position[1], env_line))

        for j in range(len(env_line)):
            #print('{} {}'.format(i, j))
            #print('inspect coord : {},{}'.format(i,j))
            #print('env: {} - view {}'.format(env_line[j], view[i][j]))
            feature = view[i][j]

            if feature is not None and ((env_line[j] == OUT_OF_GRID and feature != OUT_OF_GRID) or
                    (feature == OUT_OF_GRID and env_line[j] != OUT_OF_GRID) or
                    (env_line[j] == TREE and feature != TREE) or
                    (feature == TREE and env_line[j] != TREE)):

                #print('mismatch between env_line and line!', env_line, ' - ', view[i], ' - coord = ', position)
                return False

        #print('line ok')

    #print('all good')
    return True

#  Check whether the generated line is a valid one or not
#  Conditions: no wrong 'out of grid' positions, match line with topography, maximal number of drones not exceeded
#  Input: none features values (int list), position (int list), partial line (int list), line relative index (int),
#  current number of drones in agent's view (int), maximal number of trees/drones (int list), agent's view range (int),
#  environment (DroneCoverage)
#  Output: (bool), number of drones in the line (int), new number of visible drones (int)
def valid_line(prod, position, line, i, drone_cpt, items_limit, view_range, env):
    mid_view_range = int(view_range // 2)
    new_drone_limit = items_limit[1]
    # get line from the environment
    env_line = get_line(env, position[0] - mid_view_range + i, position[1], view_range)
    #print('Try to valid line ' + str(i) + ' - line ' + str(line) + '- product ' + str(prod) + '- position '+ str(position) + 'env_line ' + str(env_line))

    j = 0
    drone_line_cpt = 0
    grid_part_cpt = 0
    previous_feature = -1

    # loop on the line
    for idx, elm in enumerate(line):
        if elm is not None:
            feature = elm

        else:
            feature = prod[j]
            j += 1
            # specific case: central agent has a value EMPTY --> reduce the maximal number of drones (drone collision)
            if i == idx == mid_view_range and feature == EMPTY:
                #print('collision!')
                new_drone_limit -= 2

        # mismatch between environment and proposed line (focus on TREE and OUT OF GRID values)
        if ((env_line[idx] == OUT_OF_GRID and feature != OUT_OF_GRID) or
                (feature == OUT_OF_GRID and env_line[idx] != OUT_OF_GRID) or
                (feature == TREE and env_line[idx] != TREE) or
                (env_line[idx] == TREE and feature != TREE)):
            #print('mismatch between env_line and line!')
            return False, drone_line_cpt, new_drone_limit

        # exceed maximal number of drones
        if feature == DRONE:
            #print('drone found from line {} - prod {} at index i {}'.format(line, prod, i))
            # count only other drones than the one under study
            drone_line_cpt += 1
            if drone_cpt + drone_line_cpt > new_drone_limit:
                #print('there is more than 4 drones in the current view')
                return False, drone_line_cpt, new_drone_limit

        # 'out of grid' value cannot be in the middle of the grid
        if feature != OUT_OF_GRID and previous_feature in [OUT_OF_GRID, -1]:
            grid_part_cpt += 1
            if grid_part_cpt > 1:
                #print('there is a hole in the line')
                return False, drone_line_cpt, new_drone_limit

        previous_feature = feature

    #print('Valid line ' + str(i) + ' - line ' + str(line) + '- product ' + str(prod) + '- position '+ str(position) + 'env_line ' + str(env_line))
    return True, drone_line_cpt, new_drone_limit

#  Get an environment (part of) line
#  Input: environment (DroneCoverage), x,y coordinate (int), drone view range (int)
#  Output: (part of) line (int list)
def get_line(env, x, y, view_range):
    mid_range = int(view_range // 2)
    if x < 0 or x > env.nRow - 1:
        return [OUT_OF_GRID for _ in range(view_range)]

    else:
        #print('whole line {}: {}'.format(x, env.map[x]))
        return [env.map[x][y + i] if 0 <= y + i < env.nCol else OUT_OF_GRID for i in range(-mid_range, mid_range + 1)]

#  Specific product which avoids TREE coords (Generate coordinates)
#  fixed value: [x/y, value] (x: 0 / y: 1)
#  Input: list of possible feature values for x,y coordinates (int list), environment (DroneCoverage), x/y fixed
#  position (int), number of 'none' features of a coordinate (int)
#  Output: iterator of combinations of feature values (int list iterator)
def dc_coord_product(*args, env=None, fixed_value=None, repeat=1):
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]

    # x and y free
    if repeat == 2:
        #print("x and y aren't fixed")
        for pool in pools:
            #print([x for x in result])
            #print([y for y in pool])
            result = [x + [y] for x in result for y in pool if not x or (x and not (env.map[x[0]][y] == TREE))]

    # only y free
    elif not fixed_value[0]:
        #print('y is free')
        x = fixed_value[1]
        for pool in pools:
            result = [[x] + [y] for y in pool]

    # only x free
    else:
        #print('x is free')
        y = fixed_value[1]
        for pool in pools:
            result = [[x] + [y] for x in pool]

    #print('combinations of positions: {}'.format(result))
    for prod in result:
        yield tuple(prod)

#  Specific product which avoids invalid lines due to OUT_OF_GRID value (Generate a line of the view)
#  Input: list of possible feature values for a line (int list), current number of visible drones (int), maximal number
#  of trees and drones (int list), number of 'none' features of a line (int)
#  Output: iterator of combinations of feature values (int list iterator)
def dc_view_product(*args, drone_cpt, items_limit, repeat=1):
    max_tree_nb, max_drone_nb = items_limit
    pools = [tuple(pool) for pool in args] * repeat

    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool if y != DRONE or (x.count(TREE) < max_tree_nb and drone_cpt + x.count(DRONE) < max_drone_nb)]
        #print('result: {}'.format(result))

    #print('combinations of lines: {}'.format(result))
    for prod in result:
        yield tuple(prod)


#  Multiple-tasks function to update some data during the HXP process
#  Input: environment (DroneCoverage), agents (Agent list), location of the modification in the HXP process (str),
#  state-action list (int list list list-int list), additional information (dict)
#  Output: variable
def preprocess(env, agent, location, state=None, add_info=None):

    def place_agents(env, agent, state, location, dead_drone_coord=[]):
        # reset dead attribute
        if location == 'weak_paxp':
            #print('state to study: {}'.format(state))
            for a in agent:
                a.dead = False

        if location == 'hxp': actions_histo = deepcopy(state[1])

        env.init_map()
        for idx, a in enumerate(agent):
            a.env = env
            if location == 'weak_paxp':
                env.set_initPos(a, state[0][idx], state[0].count(state[0][idx]) > 1) # collision with another drone or tree
                if state[0][idx] in dead_drone_coord:
                    #print('dead man: {}'.format(state[0][idx]))
                    a.dead = True
            else:
                env.set_initPos(a, state[0][idx])

        if location == 'hxp': env.set_lastactions(actions_histo)

        env.initObs(agent)
        add_info['agent'] = agent

        return env, agent

    if location == 'copies':
        env_copy = deepcopy(env)
        agents_copy = deepcopy(agent)

        for agent in agents_copy:
            agent.dead = False

        add_info['agent'] = agents_copy

        return env_copy, agents_copy

    elif location == 'hxp':
        return place_agents(env, agent, state, location)

    elif location == 'weak_paxp':
        env, agent = place_agents(env, agent, [state[0]], location, state[1])
        del state[1] # remove info about dead attribute

        # modify state as a list of view, pos couples
        state = [ag.get_obs() for ag in agent]

        return env, agent, state

    elif location == 'impScore':
        env.init_map()
        return env, agent

    elif location == 'firstStep':
        return get_restricted_transitions(env, state), None

    elif location == 'pre_locally_minimal_paxp':
        # place agents in the map
        place_agents(env, agent, state, location)

        # extract position and view
        agent_pov_id = add_info['pred_params'][0]
        #print(agent_pov_id)
        position = state[0][agent_pov_id - 1]
        view = agent[agent_pov_id - 1].view(position)
        s = [view, position]
        print('pre Lm-PAXp process: from positions {} state of the agent {} is \n {}'.format(state, agent_pov_id - 1, s))

        # flatten state
        s = flatten_state(s)
        #print('flattened state: {}'.format(s))
        return s

    elif location == 'post_locally_minimal_paxp':
        return to_state(state[0], agent[0])

#  From state to flattened state
#  Input: state (int list list list)
#  Output: flattened state (int list)
def flatten_state(state):
    flat_state = []
    # view
    for line in state[0]:
        flat_state.extend(line)
    # position
    flat_state.extend(state[1])
    #print('flatten_state: {}'.format(flat_state))

    return flat_state

#  From flattened state to state
#  Input: flattened state (int list), agent (Agent)
#  Output: state (int list list list)
def to_state(flat_state, agent):
    view = []
    position = flat_state[-2:]
    line = []

    for feature in flat_state[:-2]:
        line.append(feature)
        if len(line) == agent.view_range:
            view.append(line)
            line = []

    return [view, position]

#  Get available actions
#  Input: states, environment (DroneCoverage)
#  Output: actions list (int list)
def get_actions(states, env):
    return [i for i in range(env.action_space.n)]

#  Extract for all agents the performed transition given a position, action and new position
#  Input: environment (DroneCoverage), positions-actions-positions list (int list-int-int list list),
#  list of previous transitions (int list)
#  Output: performed transitions (int list)
def get_restricted_transitions(env, s_a_list, previous_tr=[]):
    transitions = []
    positions, actions, new_positions = s_a_list

    for i, p in enumerate(positions):
        # Stop action
        if actions[i] == 4:
            transitions.append(0) # random transition won't impact the drone

        else:
            temp_p = env.inc(p[0], p[1], actions[i])
            #print('Temporary position: {}'.format(temp_p))
            temp_tr = []

            for tr in [0, 1, 2, 3]:
                #print(env.inc(temp_p[0], temp_p[1], tr))
                if env.inc(temp_p[0], temp_p[1], tr) == new_positions[i]:
                    temp_tr.append(tr)

            # Wind and Agent's directions are the opposite ones
            if not temp_tr or temp_p[0] in [0, env.nRow - 1] or temp_p[1] in [0, env.nCol - 1]:
                temp_tr.append((actions[i] + 2)% 4)

            # Multiple transitions are possible, we consider only one
            if not previous_tr or len(temp_tr) == 1:
                transitions.append(temp_tr[0])

            # Multiple transitions are possible, we extract the one which is not in previous_tr
            else:
                for elm in temp_tr:
                    if elm not in previous_tr:
                        transitions.append(elm)

    #print('Transitions: {}'.format(transitions))
    return transitions

### Render HXP ###

#  Display the (B-)HXP to the user
#  Input: important state-action list (state-action list), environment (DroneCoverage), agents (Agent list), importance
#  type (str), runtime (float), additional information (dict)
#  Output: None
def render(hxp, env, agent, imp_type, runtime, add_info):
    agent_id_pov = add_info['pred_params'][0]
    # Render
    for s_a_list, i in hxp:
        print("Timestep {} --- Action from drone {}:  {}".format(i, agent_id_pov, s_a_list[1][agent_id_pov - 1]))
        #env.clear_map()
        env.init_map()
        env.set_lastactions(s_a_list[1])

        for i, s in enumerate(s_a_list[0]):
            env.set_initPos(agent[i], s)

        env.initObs(agent)
        env.render(agent)

        if imp_type == 'transition':
            #env.clear_map()
            env.init_map()
            env.set_lastactions(None)
            for i, s_next in enumerate(s_a_list[2]):
                env.set_initPos(agent[i], s_next)
            env.initObs(agent)
            env.render(agent)
    # Runtime
    print("-------------------------------------------")
    print("Explanation achieved in: {} second(s)".format(runtime))
    print("-------------------------------------------")
    return

### Predicates ###

#  Get the wave range of an agent and the maximal number of highlighted cells (i.e. the maximal cover)
#  Input: agent (Agent)
#  Output: wave matrix (int list list), maximal cover (int)
def get_wave_range(agent):
    view = agent.get_obs()[0]
    #  Get only the wave range matrix
    max_cells_highlighted = (agent.wave_range * agent.wave_range - 1)
    index_range = (agent.view_range - agent.wave_range) // 2
    return [s[index_range:-index_range] for s in view[index_range:-index_range]], max_cells_highlighted

#  Check whether the agent has a perfect cover or not (local)
#  Check whether all agents have a perfect cover or not (global)
#  Input: states list (int list list list list), additional information (dict)
#  Output: (bool)
def perfect_cover(states, info):
    agents = info['agent']
    agent_id_pred = info['pred_params'][1]
    if states:
        agents = set_multiple_obs(states, agents)

    # Check perfect cover for one agent (pred_type < n + 1)
    if agent_id_pred:
        sub_view, max_highlighted = get_wave_range(agents[agent_id_pred - 1])
        #  Another drone in range or a tree in coverage area zone
        return sum([sub_list.count(1) for sub_list in sub_view]) == max_highlighted

    # Check perfect cover for all agents
    else:
        # If at least one drone has an imperfect cover, the configuration is ignored
        for idx, agent in enumerate(agents):
            if states:
                agent.set_obs(states[idx])

            tmp_info = {'agent': [agent], 'pred_params':[None, 1]}
            if not crash([], tmp_info) and imperfect_cover([], tmp_info):
                return False

        return True

#  Check whether the agent has an imperfect cover or not (local)
#  Check whether all agents have an imperfect cover or not (global)
#  Input: states list (int list list list list), additional information (dict)
#  Output: (bool)
def imperfect_cover(states, info):
    agents = info['agent']
    agent_id_pred = info['pred_params'][1]
    if states:
        agents = set_multiple_obs(states, agents)

    # Check imperfect cover for one agent
    if agent_id_pred:
        sub_view, max_highlighted = get_wave_range(agents[agent_id_pred - 1])
        #  Another drone in range or a tree in coverage area zone
        return sum([sub_list.count(1) for sub_list in sub_view]) != max_highlighted

    # Check imperfect cover for all agents
    else:
        # If at least one drone has a perfect cover, the configuration is ignored
        for idx, agent in enumerate(agents):
            if states:
                agent.set_obs(states[idx])

            tmp_info = {'agent': [agent], 'pred_params': [None, 1]}
            if not crash([], tmp_info) and perfect_cover([], tmp_info):
                return False

        return True

#  Check whether the agent is crashed or not (local)
#  Check whether all agents are crashed or not (global)
#  Input: states list (int list list list list), additional information (dict)
#  Output: (bool)
def crash(states, info):
    agents = info['agent']
    agent_id_pred = info['pred_params'][1]
    if states:
        agents = set_multiple_obs(states, agents)

    # Check crash for one agent
    if agent_id_pred:
        view = agents[agent_id_pred - 1].get_obs()[0]
        return view[len(view) // 2][len(view[0]) // 2] != 2

    # If at least one drone is not crashed, the configuration is ignored
    else:
        for idx, agent in enumerate(agents):
            if states:
                agent.set_obs(states[idx])
            tmp_info = {'agent': [agent], 'pred_params': [None, 1]}
            if not crash([], tmp_info):
                return False

        return True

#  Check whether the agent has no drones in its view range or not (local)
#  Check whether all agents have no drones in their view range or not (global)
#  Input: states list (int list list list list), additional information (dict)
#  Output: (bool)
def no_drones(states, info):
    agents = info['agent']
    agent_id_pred = info['pred_params'][1]
    if states:
        agents = set_multiple_obs(states, agents)

    # Check no_drones for one agent
    if agent_id_pred:
        view = agents[agent_id_pred - 1].get_obs()[0]
        return sum([sub_list.count(2) for sub_list in view]) == 1

    # If at least one drone is in another drone's neighborhood, the configuration is ignored
    else:
        for idx, agent in enumerate(agents):
            if states:
                agent.set_obs(states[idx])

            tmp_info = {'agent': [agent], 'pred_params': [None, 1]}
            if not crash([], tmp_info) and not no_drones([], tmp_info):
                return False

        return True

#  Check whether the agent has the maximal reward or not (local)
#  Check whether all agents have the maximal reward or not (global)
#  Input: states list (int list list list list), additional information (dict)
#  Output: (bool)
def max_reward(states, info):
    agents = info['agent']
    agent_id_pred = info['pred_params'][1]
    if states:
        agents = set_multiple_obs(states, agents)

    # Check max_reward for one agent
    if agent_id_pred:
        agent = agents[agent_id_pred - 1]
        tmp_info = {'agent': [agent], 'pred_params': [None, 1]}
        return not crash([], tmp_info) and perfect_cover([], tmp_info) and no_drones([], tmp_info)

    # If at least one drone has not the max reward, the configuration is ignored
    else:
        for idx, agent in enumerate(agents):
            if states:
                agent.set_obs(states[idx])

            tmp_info = {'agent': [agent], 'pred_params': [None, 1]}
            if not max_reward([], tmp_info):
                return False

        return True

#  Check whether the agent is in a specific region or not (local)
#  Check whether all agents are in a distinct region or not (global)
#  Input: states list (int list list list list), additional information (dict)
#  Output: (bool)
def region(states, info):
    agents = info['agent']
    agent_id_pov = info['pred_params'][0]
    idx_region = info['pred_params'][1]
    #print('agent_id_pov', agent_id_pov)

    if states:
        agents = set_multiple_obs(states, agents)

    # Check a region for an agent
    if idx_region:

        agent = agents[agent_id_pov - 1]
        x, y = agent.get_obs()[1]
        #print('coords: ', x, y)
        limit = agent.env.nRow // 2
        #  Map divided into four parts
        if x < limit and y < limit:
            r = 1
        elif x >= limit > y:
            r = 3
        elif x < limit <= y:
            r = 2
        else:
            r = 4
        return idx_region == r

    # Each drone is in a distinct region
    else:
        regions = []
        for idx, agent in enumerate(agents):

            if states:
                agent.set_obs(states[idx])

            tmp_info = {'agent': [agent], 'pred_params': [None, 1]}

            if not crash([], tmp_info):
                for r in [1, 2, 3, 4]:

                    tmp_info['pred_params'] = [1, r]
                    if region([], tmp_info):
                        if r in regions:
                            return False
                        else:
                            regions.append(r)
        return True

#  Check whether a predicate defined via PAXp holds or not
#  Input: state (int list list list), additional information (dict)
#  Output: (bool)
def redefined_predicate(s, info):
    view_pred, position_pred = info['redefined_predicate']
    agent_id_pov = info['pred_params'][0]
    view, position = s[agent_id_pov - 1]

    # Check position
    for idx, coord in enumerate(position_pred):
        if coord is not None and coord != position[idx]:
            #print('difference pred {} -- state {}'.format(coord, position[idx]))
            return False

    # Check view
    for i in range(len(view_pred)):
        for j in range(len(view_pred[0])):

            feature = view_pred[i][j]
            if feature is not None and feature != view[i][j]:
                #print('difference pred {} -- state {}'.format(feature, view[i][j]))
                return False

    #print('all good')
    return True

### Find histories for a specific predicate ###

#  Verify if the last state from a proposed history respects a predicate
#  Input: states list (int list list list list), predicate (str), additional information (dict)
#  Output: (bool)
def valid_history(states, predicate, info):
    agents = info['agent']
    agent_id_pred = info['pred_params'][1]
    #print(predicate)
    #print(info['pred_params'])

    if states:
        agents = set_multiple_obs(states, agents)

    # Perfect cover predicate
    if predicate == 'perfect_cover':
        if agent_id_pred:
            tmp_info = {'agent': [agents[agent_id_pred - 1]], 'pred_params': [None, 1]}
            return not crash([], tmp_info) and perfect_cover([], tmp_info)

        else:
            return perfect_cover(states, info)

    # Imperfect cover predicate
    elif predicate == 'imperfect_cover':
        if agent_id_pred:
            tmp_info = {'agent': [agents[agent_id_pred - 1]], 'pred_params': [None, 1]}
            return not crash([], tmp_info) and imperfect_cover([], tmp_info)

        else:
            return imperfect_cover(states, info)

    # Max reward predicate
    elif predicate == 'max_reward':
        if agent_id_pred:
            tmp_info = {'agent': [agents[agent_id_pred - 1]], 'pred_params': [None, 1]}
            return max_reward([], tmp_info)

        else:
            return max_reward(states, info)

    # No drones predicate
    elif predicate == 'no_drones':
        if agent_id_pred:
            tmp_info = {'agent': [agents[agent_id_pred - 1]], 'pred_params': [None, 1]}
            return not crash([], tmp_info) and no_drones([], tmp_info)

        else:
            return no_drones(states, info)

    # Crash predicate
    elif predicate == 'crash':
        tmp_info = {'agent': [agents[agent_id_pred - 1]], 'pred_params': [None, 1]}
        if agent_id_pred:
            return crash([], tmp_info)

        else:
            return crash(states, info)

    # Region predicate
    elif predicate == 'region':
        #print('valid_history')
        agent_id_pov = info['pred_params'][0]
        # agent_id_pred is the region to study
        if agent_id_pred:

            tmp_info = {'agent': [agents[agent_id_pov - 1]], 'pred_params': [None, 1]}
            if not crash([], tmp_info):
                tmp_info = {'agent': [agents[agent_id_pov - 1]], 'pred_params': [1, agent_id_pred]}
                return region([], tmp_info)
            else:
                return False

        else:
            return region(states, info)

#  Set a state to each agent
#  Input: states list (int list list list list), agents (Agent list)
#  Output: agents (Agent list)
def set_multiple_obs(states, agents):
    for i, a in enumerate(agents):
        a.set_obs(states[i])

    return agents

#  Get exhaustive list of transitions
#  Input: environment (DroneCoverage), type of transition function (str), number of agents (int)
#  Output: list of exhaustive transitions for nb_agents - 1 and nb_agents (transitions list)
def get_transition_exhaustive_lists(env, tr_func, nb_agents):
    return [list(product(env.P, repeat=nb_agents - 1)), list(product(env.P, repeat=nb_agents))] if tr_func != 'approx' else []
