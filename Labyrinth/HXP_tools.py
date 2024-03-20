import random
from copy import deepcopy

###  HXP functions ###

#  Get from a state-action couple, the entire/part of the transitions available, i.e. the new states associated with
#  their probabilities
#  Input: agent's state (int), action (int), environment (Labyrinth), importance score method (str), number of
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

#  Compute the argmax of an array
#  Input: an array (list)
#  Output: index of the maximum value (int)
def argmax(array):
    array = list(array)
    return array.index(max(array))

#  Check whether a state is terminal or not
#  Input: state (int), environment (Labyrinth), additional information (dictionary)
#  Output: (bool)
def terminal(s, env, add_info):
    #  Get the current position depending on the agent's type
    row, col = s // len(env.map[0]), s % len(env.map[0])
    return env.map[row][col] in ['S', 'P']

#  Multiple-tasks function to update some data during the (B-)HXP process
#  Input: environment (Labyrinth), agent, location of the modification in the HXP process (str),
#  state-action list (int list), additional information (dictionary)
#  Output: variable
def preprocess(env, agent, location, s_a_list=None, add_info=None):
    return env, agent

#  Check whether an importance score can be computed or not
#  Input: action (int), importance type (str), additional information (dictionary)
#  Output: (bool)
def constraint(action, imp_type, add_info):
    return False

#  Get available actions from a state
#  Available actions are similar no matter the state
#  Input: state (int), environment (Labyrinth)
#  Output: action list (int list)
def get_actions(s, env):
    return [i for i in range(env.actions)]

#  Render the most important action(s) / transition(s)
#  Input: state action list to display (int list), environment (Labyrinth), agent,
#  importance type (str), runtime (float), additional information (dictionary)
#  Output: None
def render(hxp, env, agent, imp_type, runtime, add_info):
    # Render
    env_copy = deepcopy(env)
    for s_a_list, i in hxp:
        print("Timestep {}".format(i))
        env_copy.set_obs(s_a_list[0])
        env_copy.render()
        print("    ({})".format(["Left", "Down", "Right", "Up"][s_a_list[1]]))
        if imp_type == 'transition':
            env_copy.set_obs(s_a_list[2])
            env_copy.render()

    # Runtime
    print("-------------------------------------------")
    print("Explanation achieved in: {} second(s)".format(runtime))
    print("-------------------------------------------")
    return

### Backward HXP functions ###

#  Sample n states from the state space. These states match the set of fixed features of v. TODO
def sample(env, v, i, n=0):
    pass

### Predicates ###

#  Check whether the agent reach the exit of the labyrinth or not
#  Input: agent's state (int), additional information (dictionary)
#  Output: bool
def reach_exit(s, info):
    env = info['env']
    #  Get the current position depending on the agent's type
    row, col = s // len(env.map[0]), s % len(env.map[0])
    return env.map[row][col] == 'S'
