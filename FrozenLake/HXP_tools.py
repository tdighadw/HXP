import random
from copy import deepcopy

### HXP functions ###

#  Get from a state-action couple, the entire/part of the transitions available, i.e. the new states associated with
#  their probabilities
#  Input: agent's state (int (list)), action (int), environment (MyFrozenLake), importance score method (str), number of
#  exhaustive/deterministic steps (int), additional information (dictionary), importance type (str)
#  Output: list of transition-probability couples (couple list)
def transition(s, a, env, approx_mode, exh_steps=0, det_tr=0, add_info=None, imp_type=None):
    transitions = [(t[0], t[1]) for t in deepcopy(env.P[s][a])]

    # Look all possible transitions from s
    if approx_mode == 'none' or exh_steps:
        return transitions

    else:
        # Look for the most probable transition
        if approx_mode == 'last':
            # Specific case: equiprobable transitions
            return extract_transitions(1, transitions, approx_mode)
        # Select the 'det' most probable transition(s)
        else:
            return extract_transitions(det_tr, transitions, approx_mode)

#  Check whether the state is terminal or not
#  Input: state (int), environment (MyFrozenLake), additional information (dict)
#  Output: (bool)
def terminal(s, env, add_info):
    state = s if not env.many_features else s[0]
    row, col = state // env.nCol, state % env.nCol

    return bytes(env.desc[row, col]) in b"GH"

#  Compute the argmax of an array
#  Input: an array (list)
#  Output: index of the maximum value (int)
def argmax(array):
    array = list(array)
    return array.index(max(array))

#  Extract n most probable transitions
#  Input: number of transition to extract (int), transitions (tuple list), importance score method (str)
#  Output: most probable transition(s) (tuple list)
def extract_transitions(n, transitions, approx_mode):
    most_probable = []

    while n != len(most_probable):
        probas = [t[0] for t in transitions]
        max_pr, idx_max_pr = max(probas), argmax(probas)
        tmp_cpt = probas.count(max_pr)
        # Only one transition is the current most probable one
        if tmp_cpt == 1:
            temp_t = list(transitions[idx_max_pr])
            most_probable.append(temp_t)
            transitions.remove(transitions[idx_max_pr])

        else:
            # There are more transitions than wanted (random pick)
            if tmp_cpt > n - len(most_probable):
                random_tr = random.choice([t for t in transitions if t[0] == max_pr])
                #print('random_tr: {}'.format(random_tr))
                temp_random_tr = list(random_tr)
                most_probable.append(temp_random_tr)
                transitions.remove(random_tr)

            else:
                tmp_list = []
                for t in transitions:
                    if t[0] == max_pr:
                        temp_t = list(t)
                        most_probable.append(temp_t)
                        tmp_list.append(t)
                for t in tmp_list:
                    transitions.remove(t)

    # Probability distribution
    sum_pr = sum([p for p, s in most_probable])
    if sum_pr != 1.0:
        delta = 1.0 - sum_pr
        add_p = delta / len(most_probable)
        for elm in most_probable:
            elm[0] += add_p
    return most_probable

#  Multiple-tasks function to update some data during the HXP process
#  Input: environment (MyFrozenLake), agent (Agent), location of the modification in the HXP process (str),
#  state-action list (int (list)-int list), additional information (dictionary)
#  Output: variable
def preprocess(env, agent, location, s_a_list=None, add_info=None):
    #  Convert tuple to list for paxp
    if location == 'pre_locally_minimal_paxp':
        return list(s_a_list[0])
    #  Convert list to tuple for paxp
    elif location == 'post_locally_minimal_paxp':
        return tuple(s_a_list[0])
    elif location == 'weak_paxp':
        return env, agent, s_a_list

    return env, agent

#  Check whether an importance score can be computed or not
#  Input: action (int), importance type (str), additional information (dictionary)
#  Output: (bool)
def constraint(action, imp_type, add_info):
    return False

#  Get available actions from a state (available actions are similar no matter the state)
#  Input: state (int list), environment (MyFrozenLake)
#  Output: action list (int list)
def get_actions(s, env):
    return list(env.P[s].keys())

#  Render the most important action(s) / transition(s)
#  Input: state-action list to display (int (list)-int  list), environment (MyFrozenLake), agent (Agent),
#  importance type (str), runtime (float), additional information (dictionary)
#  Output: None
def render(hxp, env, agent, imp_type, runtime, add_info):
    # Render
    env_copy = deepcopy(env)
    for s_a_list, i in hxp:
        print("Timestep {}".format(i))
        env_copy.setObs(s_a_list[0])
        env_copy.render()
        print("    ({})".format(["Left", "Down", "Right", "Up"][s_a_list[1]]))
        if imp_type == 'transition':
            env_copy.setObs(s_a_list[2])
            env_copy.render()
    # Runtime
    print("-------------------------------------------")
    print("Explanation achieved in: {} second(s)".format(runtime))
    print("-------------------------------------------")
    return

### Backward HXP functions ###

#  Sample n states from the state space. These states match the set of fixed features of v.
#  This is an exhaustive sampling.
#  Input: environment (MyFrozenLake), partial state (int (list)), index of feature to remove from v (int),
#  number of samples to generate (int), additional information (dict)
#  Output: list of states (int (list) list)
def sample(env, v, i, n, add_info=None):
    feature_value = v[i]
    v[i] = None
    allow_terminal = True
    print('sample state which match v: {}'.format(v))
    states = list(env.P.keys())
    samples = []

    for state in states:
        # match between state and set of fixed features of v
        if valid(state, v) and (allow_terminal or not terminal(state, env, None)) :
            samples.append(state)
            if len(samples) == n:
                break

    #print('len(samples): {}'.format(len(samples)))
    v[i] = feature_value
    return samples

#  Check whether a point match with the fixed features of v or not
#  Input: state (int (list)), partial state (int (list))
#  Output: (bool)
def valid(p, v):
    for idx, value in enumerate(v):
        # the feature is fixed, v and p don't share same value
        if value is not None and value != p[idx]:
            return False
    return True

### Predicates ###

#  Check whether the agent reaches the goal or not
#  Input: state (int (list)), additional information (dictionary)
#  Output: (bool)
def win(s, info):
    env = info['env']
    state = s if not env.many_features else s[0]
    win_state = env.to_s(env.goal_position[0], env.goal_position[1])

    return state == win_state

#  Check whether the agent avoids falling into a hole or not
#  Input: state (int (list)), additional information (dictionary)
#  Output: (bool)
def avoid_holes(s, info):
    env = info['env']
    state = s if not env.many_features else s[0]
    holes = get_holes(env)

    return state not in holes

#  Check whether the agent falls into a hole or not
#  Input: state (int (list)), additional information (dictionary)
#  Output: (bool)
def holes(s, info):
    env = info['env']
    state = s if not env.many_features else s[0]
    holes = get_holes(env)
    return state in holes

#  Check whether the agent is in a specific cell of the map or not
#  Input: state (int (list)), additional information (dictionary)
#  Output: (bool)
def specific_state(s, info):
    params = info['pred_params']
    env = info['env']
    state = s if not env.many_features else s[0]

    return state == params[0]

#  Check whether the agent is in a specific region of the map or not
#  Input: state (int (list)), additional information (dictionary)
#  Output: (bool)
def specific_part(s, info):
    params = info['pred_params']
    env = info['env']
    state = s if not env.many_features else s[0]

    return state in params[0]

#  Check whether a predicate defined via PAXp holds or not
#  Input: state (int (list)), additional information (dictionary)
#  Output: (bool)
def redefined_predicate(s, info):
    predicate = info['redefined_predicate']
    for idx, feature in enumerate(predicate):
        if feature is not None and feature != s[idx]:
            return False
    return True

#  Extract the list of holes in the map
#  Input: environment (MyFrozenLake)
#  Output: list of holes (int list)
def get_holes(env):
    holes = []
    for i in range(len(env.desc)):
        for j in range(len(env.desc[0])):
            if bytes(env.desc[i, j]) in b"H":
                holes.append(env.to_s(i, j))
    return holes

### Find histories for a specific predicate ###

#  Verify if the last state from a proposed history respects a predicate
#  Input: state (int (list)), predicate (str), additional information (int)
#  Output: respect or not of the predicate (bool)
def valid_history(s, predicate, info):
    if predicate == 'win':
        return win(s, info)
    elif predicate == 'holes':
        return holes(s, info)
    elif predicate == 'specific_state':
        return specific_state(s, info)
    elif predicate == 'specific_part':
        return specific_part(s, info)
