import csv
import sys
import time
from copy import deepcopy
from DynamicObstacles.HXP_tools import hxp_render

import numpy as np

#  Argmax function
def argmax(array):
    array = list(array)
    return array.index(max(array))

#  Select the state used to get the new predicate
#  Input: state/action data list, history indexes (int list), state selection method (str)
#  Output: state utility (float), state, action/transition associate to the selected state, state index (int)
def select(data, indexes, method):
    # Get the first element in the sub-sequence
    if method == 'first':
        best_elm_idx = 0
    # Get the element with the higher importance score
    else:
        best_elm_idx = argmax([d[0] for d in data])

    print('best_elm_idx: ', best_elm_idx)
    # round operation to avoid float accuracy issue
    return round(data[best_elm_idx][1], 5), data[best_elm_idx][2], indexes[best_elm_idx], len(indexes) - best_elm_idx - 1

class HXP:

    def __init__(self, name, agent, env, predicates, functions, add_info, context='mono'):
        self.name = name
        self.context = context
        self.agent = agent
        self.env = env
        # dictionary of couple (name,  (params, function)) of each predicate
        self.predicates = predicates
        self.add_info = add_info
        # functions
        transition, terminal, constraint, sample, preprocess, get_actions, render = functions
        self.transition = transition
        self.terminal = terminal
        self.constraint = constraint
        self.sample = sample
        self.preprocess = preprocess
        self.get_actions = get_actions
        self.render = render

    #  Explain a history using whether HXP or B-HXP
    #  Input: explanation mode (user/no_user/compare) (str), history (state-action list), predicate (str),
    #  importance score method (str list), number of actions to highlight for HXP computation (int), element to
    #  explain (str), csv file for utilities storage (str), delta and l values for B-HXP computation (int/float list),
    #  coefficients to balance between initial and redefined predicate (float list)
    #  Output: explanations, computed importance scores (float list list), (B-)HXP runtimes (float list),
    #  studied predicates (str list list)
    def explain(self, mode, history, predicate=None,  approaches=[], n=2, imp_type='action', csv_file='utilities.csv', backward=[], coefs=[]):
        HXPs = []
        Imp_scores = []
        Runtimes = []
        Predicates = []
        H = list(history.queue)

        # User mode
        if mode == 'user':
            answer = False
            good_answers = ["yes", "y"]
            while not answer:
                # Action ------------------------------------------------
                question = "Do you want an action HXp?"
                action_hxp = input(question)
                if action_hxp in good_answers:
                    keys = self.predicates.keys()
                    predicate_question = "Which predicate do you want to test? (" + "/".join(keys) + ") "
                    pred, pred_params = self.extract(input(predicate_question))
                    self.add_info['pred_params'] = pred_params
                    # Compute an (approx.) HXP or B-HXP
                    hxp, imp_scores, runtime, predicates = self.hxp(H, pred, approaches[0], n,
                                                       'action', csv_file, backward, coefs)
                    HXPs.append(hxp)
                    Imp_scores.append(imp_scores)
                    Runtimes.append(runtime)
                    Predicates.append(predicates)
                    # Render
                    self.render(hxp, self.env, self.agent, 'action', Runtimes[0], self.add_info)
                # Transition -------------------------------------------
                question = "Do you want an transition HXp?"
                transition_hxp = input(question)
                if transition_hxp in good_answers:
                    keys = self.predicates.keys()
                    predicate_question = "Which predicate do you want to test? (" + "/".join(keys) + ") "
                    pred, pred_params = self.extract(input(predicate_question))
                    self.add_info['pred_params'] = pred_params
                    # Compute an (approx.) HXP or B-HXP
                    hxp, imp_scores, runtime, predicates = self.hxp(H, pred, approaches[0], n,
                                                       'transition', csv_file, backward, coefs)
                    HXPs.append(hxp)
                    Imp_scores.append(imp_scores)
                    Runtimes.append(runtime)
                    Predicates.append(predicates)
                    # Render
                    self.render(hxp, self.env, self.agent, 'transition', Runtimes[0], self.add_info)
                answer = True
        else:
            pred, pred_params = self.extract(predicate)
            self.add_info['pred_params'] = pred_params

            # No user mode
            if mode == 'no_user':
                # Compute an (approx.) HXP or B-HXP
                hxp, imp_scores, runtime, predicates = self.hxp(H, pred, approaches[0], n,
                                                   imp_type, csv_file, backward, coefs)
                HXPs = hxp
                Imp_scores = imp_scores
                Runtimes.append(runtime)
                Predicates = predicates
                # Render
                self.render(hxp, self.env, self.agent, imp_type, Runtimes[0], self.add_info)

            # Compare mode
            else:
                # Compute an (approx.) HXP or B-HXP
                for approach in approaches:
                    hxp, imp_scores, runtime, predicates = self.hxp(H, pred, approach, n,
                                                       imp_type, csv_file, backward, coefs)
                    HXPs.append(hxp)
                    Imp_scores.append(imp_scores)
                    Runtimes.append(runtime)
                    Predicates.append(predicates)
                # Display runtimes
                print('------ Runtimes -------')
                for i, r in enumerate(Runtimes):
                    print('Approach: {} -- Runtime: {}'.format(approaches[i], r))
                print('-----------------------')

        return HXPs, Imp_scores, Runtimes, Predicates

    #  Compute an HXP or B-HXP of a given history
    #  Input: history (state-action list), predicate (str), importance score method (str), number of actions to
    #  highlight for HXP computation (int), element to explain (str), csv file for utilities storage (str), delta and l
    #  values for B-HXP computation (int/float list),
    #  coefficients to balance between initial and redefined predicate (float list)
    #  Output: explanation, computed importance scores (float list), (B-)HXP runtime (float),
    #  studied predicates (str list)
    def hxp(self, H, predicate, approach, n, imp_type, csv_file, backward, coefs):
        # Init
        hxp = []
        data = []
        predicates =[]
        z_scores = []
        first_predicate = predicate
        select_method = self.add_info['select']
        horizon = backward[0] if backward else 0
        delta = backward[1] if backward else 0
        k = ((len(H) - 1) // 2) - 1 if not backward else horizon - 1
        print('Scenarios length: {}'.format(k))

        env_copy, agent_copy = self.preprocess(self.env, self.agent, 'copies', None, self.add_info) # e.g. deep copies

        start_time = time.perf_counter()

        # HXP
        if not horizon:
            for i in range(0, len(H) - 1, 2):

                # Compute importance score of i^th action in the history
                # Similar horizon for each action: k / Decreasing horizon: k - (i//2)
                self.evaluate(H, i, env_copy, agent_copy, imp_type, data, approach, predicate, k, csv_file, coefs)

        # B-HXP
        else:
            top_cur_idx = []
            predicates.append(predicate)
            i_max = len(H) - 1
            i_min = -1

            while i_min != 0:
                i_min = max(0, i_max - (horizon * 2))
                tmp_data = []
                indexes = [i for i in range(i_min, i_max, 2)]

                for j, i in enumerate(indexes):

                   # Compute importance score of i^th action in the history
                   # Similar horizon for each action: k / Decreasing horizon: len(indexes) - j - 1
                   self.evaluate(H, i, env_copy, agent_copy, imp_type, tmp_data, approach, predicate, k, csv_file, coefs)

                # Select the best action/transition
                utility, s_a_list, idx, h = select(tmp_data, indexes, select_method)
                print('selected state {} - utility {} - idx: {}'.format(s_a_list[0], utility, idx))

                # Early stop: utility of an action is 0.0
                if not utility:
                    i_min = 0
                    # Update idx since we don't explore the whole history
                    if select_method == 'first':
                        for i in range(len(top_cur_idx)):
                            top_cur_idx[i][1] -= idx
                    top_cur_idx.append([idx, 0])
                else:
                    z_scores.append(utility)
                # Define new predicate
                if i_min:

                    # Similar horizon for each action: k / Decreasing horizon: len(indexes) - j - 1
                    predicate = self.locally_minimal_paxp(s_a_list, utility, predicate, k, delta, imp_type, approach, coefs)

                    # Early stop: predicate is an empty PAXp
                    if self.is_none_predicate(predicate):
                        i_min = 0
                    else:
                        predicates.append(predicate)
                        # Store new predicate to use it in the future utility computations
                        self.add_info['redefined_predicate'] = predicate
                        predicate = 'redefined_predicate' if not coefs else ['redefined_predicate', first_predicate]

                # updates (condition related to "Early stop: utility of an action is 0.0")
                if not top_cur_idx or top_cur_idx[-1][1]:
                    top_cur_idx.append([idx, idx])
                    i_max = idx

                tmp_data.extend(data)
                data = tmp_data

        # Rank the scores to select the n most important action
        print('Approach: {}'.format(approach))
        tmp_list = [elm[0] for elm in data]
        print('Not rounded scores: {}'.format(tmp_list))
        imp_scores = [round(item, 5) for item in tmp_list]
        print("Scores : {}".format(imp_scores))

        # HXP (extract indexes using importance scores)
        if not horizon:
            top_idx = np.sort(np.argpartition(np.array(tmp_list), -n)[-n:])
            current_idx = top_idx
        # B-HXP (indexes are already available)
        else:
            top_idx = [e[0] // 2 for e in top_cur_idx]
            top_idx.reverse()
            current_idx = top_idx #[e[1] // 2  for e in top_cur_idx]
            n = len(top_cur_idx)

        print("Current index(es): {}".format(current_idx))
        for i in range(n):
            hxp.append([(H[current_idx[i] * 2], H[current_idx[i] * 2 + 1]), top_idx[i]])
        final_time = time.perf_counter() - start_time

        # Additional info when a B-HXP is computed
        if predicates:
            predicates.reverse()
            z_scores.reverse()
            print('Predicates studied along the search:')
            for p in predicates: print(p)
            print('Z scores: {} - {}'.format([round(z, 3) for z in z_scores], z_scores))

        return hxp, imp_scores, final_time, predicates

    #  Compute the importance score of the action pi(s) from s
    #  Input: state-action list (state-action list), scenarios length (int), importance score method (str), environment
    #  (Environment), agent(s) list (Agent list), actions list (int list), importance type (str), predicate (str), file
    #  name (str), coefficients to balance between initial and redefined predicate (float list)
    #  Output: importance score of the action pi(s) from s (float), utility of pi(s) from s (float)
    def imp_score(self, s_a_list, k, approach, env, agent, actions, imp_type, predicate, csv_file, coefs):
        # Get two initial sets
        S, S_not = self.first_step(s_a_list, actions, env, agent, imp_type, predicate)

        # Generate scenarios
        env, agent = self.preprocess(env, agent, 'impScore', s_a_list, self.add_info)
        print('Start scenarios generation for S')
        S = self.succ(S, k, approach, env, agent, imp_type, predicate)

        env, agent = self.preprocess(env, agent, 'impScore', s_a_list, self.add_info)
        print('Start scenarios generation for S_not')
        S_not = self.succ(S_not, k, approach, env, agent, imp_type, predicate)

        # Compute utilities
        if self.add_info.get('initial_s'):
            self.add_info['initial_s'] = s_a_list[0]

        # If B-HXP is computed, the predicate to study is changed (test new feature: two predicates are studied)
        pred = predicate #if not self.add_info.get('redefined_predicate') else 'redefined_predicate' if not coefs else ['redefined_predicate', predicate]
        utilities = self.u(S, S_not, env, agent, pred, coefs)

        # Determine importance score
        imp_score = utilities[0] - utilities[-1]

        # Store/Display info
        self.store(csv_file, s_a_list, utilities, pred, approach)
        return imp_score, utilities[0]

    #  Generate the scenarios (in an exhaustive or approximate way)
    #  Input: Set of states obtained after 1 step starting by doing an action from s (state-probability list),
    #  scenarios length (int), importance score method (str), environment (Environment), agent(s) list (Agent (list)),
    #  importance type (str), predicate (str)
    #  Output: Set of states obtained after k+1 steps starting by doing an action from s (state-probability list)
    def succ(self, S, k, approach, env, agent, imp_type, predicate):
        S_tmp = []
        fixed_horizon = self.add_info['fixed_horizon']

        if approach != 'exh':
            det_transition = int(approach.split('_')[1])
            approx_mode = approach.split('_')[0]
        else:
            det_transition = 0
            approx_mode = 'none'

        # Limit the number of last det. transition to the depth of scenarios
        if approx_mode == 'last' and det_transition > k:
            det_transition = k

        # print(approach)
        # print('det tr: {} -- approx_mode: {}'.format(det_transition, approx_mode))

        # Determine the number of exhaustive step(s)
        exh_steps = k if approx_mode == 'none' else k - det_transition if approx_mode == 'last' else 0

        # Generate scenarios
        for _ in range(k):

            if all(isinstance(el, list) for el in S):
                # print('len(S[0]): {}'.format(len(S[0])))

                for sublist in S:

                    # print('sublist --------')
                    S_tmp_sublist = []
                    for s in sublist:
                        # Extract state, proba
                        state, proba, respect = s

                        if not self.is_terminal(state, env) and (fixed_horizon or not respect):
                            action = self.predict(agent, state)
                            #print('from state: {} -- action: {}'.format([s[1] for s in state], action))

                            for p, new_s in self.transition(state, action, env, approx_mode, exh_steps, det_transition, self.add_info, imp_type):
                                # Succession of probabilities
                                if fixed_horizon:
                                    S_tmp_sublist.append((new_s, proba * p, 0))
                                else:
                                    _, r = self.u([(new_s, 1.0, 0)], [], env, agent, predicate)
                                    #if r: print('new branch stopped: {} with pred: {} - proba {}'.format(new_s, predicate, proba * p))

                                    S_tmp_sublist.append((new_s, proba * p, r))
                        else:
                            #if respect: print('already stoped branch: {} - proba: {}'.format(state, proba))

                            # Add the terminal state
                            S_tmp_sublist.append((state, proba, respect))

                    S_tmp.append(S_tmp_sublist)

                S = S_tmp
                S_tmp = []

            # Used for generating the list of reachable final states from the current action/transition
            else:
                # print('len(S): {}'.format(len(S)))
                for s in S:
                    # Extract state, proba
                    state, proba, respect = s

                    if not self.is_terminal(state, env) and (fixed_horizon or not respect):
                        action = self.predict(agent, state)
                        # print('from state: {} -- action: {}'.format([s[1] for s in state], action))

                        for p, new_s in self.transition(state, action, env, approx_mode, exh_steps, det_transition, self.add_info, imp_type):
                            # Succession of probabilities
                            if fixed_horizon:
                                S_tmp.append((new_s, proba * p, 0))
                            else:
                                _, r = self.u([(new_s, 1.0, 0)], [], env, agent, predicate)
                                #if r: print('new branch stopped: {} - pred: {} - proba {}'.format(new_s, predicate, proba * p))
                                S_tmp.append((new_s, proba * p, r))
                    else:
                        # Add the terminal state
                        S_tmp.append((state, proba, respect))

                S = S_tmp
                S_tmp = []

            exh_steps -= 1 if exh_steps else 0

        return S

    #  Compute the utility (probability to reach a state k time-step later which respects the predicate starting from s)
    #  Input: Set of states obtained after k steps starting by doing pi(s) from s (state-probability list), Set of
    #  states obtained after k steps starting by doing all a' != pi(s) from s (state-probability list list),
    #  environment (Environment), agent(s) list (Agent (list)), predicate (str),
    #  coefficients to balance between initial and redefined predicate (float list)
    #  Output: list of utilities: utility to perform a from s (float),
    #  list of utilities / best utility / average utility not to perform a from s (float list / float / float)
    def u(self, S, S_not, env, agent, predicate, coefs):
        # Dictionary of parameters
        info = deepcopy(self.add_info)
        info['env'] = env
        info['agent'] = agent

        # Setting: two predicates are studied simultaneously
        if not isinstance(predicate, str):
            print('two predicates: ', predicate)

        u = 0.0 if isinstance(predicate, str) else [0.0, 0.0]
        # print('len(S): {}'.format(len(S)))
        # print('len(S_not[0]): {} '.format(len(S_not[0])))

        for (s, p, r) in S:
            #print(s)
            if isinstance(predicate, str):
                if self.predicates[predicate](s, info):
                    r = 1
                    u += p
            # Setting: two predicates are studied simultaneously
            else:
                r = [0, 0]
                for i, pred in enumerate(predicate):
                    if self.predicates[pred](s, info):
                        r[i] = 1
                        u[i] += p

        # Early stop: it's used for the locally minimal paxp computation, where we only need utility
        # It's also used to compute the respect for the predicate for one state during the scenario search
        if not S_not:
            # Setting: two predicates are studied simultaneously
            if not isinstance(predicate, str):
                print('use both predicates')
                u = coefs[0] * u[0] + coefs[1] * u[1]
            return u, r
        # Utility of all other actions available from s
        u_not = []
        for sublist in S_not:
            tmp_u = 0.0 if isinstance(predicate, str) else [0.0, 0.0]
            for (s, p, r) in sublist:
                if isinstance(predicate, str):
                    if self.predicates[predicate](s, info):
                        tmp_u += p
                        r = 1
                else:
                    r = [0, 0]
                    for i, pred in enumerate(predicate):
                        if self.predicates[pred](s, info):
                            r[i] = 1
                            tmp_u[i] += p

            u_not.append(tmp_u)

        print('u_not: {}'.format(u_not))
        print('u: {}'.format(u))

        # Setting: two predicates are studied simultaneously
        if not isinstance(predicate, str):
            u = coefs[0] * u[0] + coefs[1] * u[1]
            u_not = [coefs[0] * u_n[0] + coefs[1] * u_n[1] for u_n in u_not]
            print('u after coef: {}'.format(u))
            print('u_not after coef: {}'.format(u_not))

        # Max contrastive utility
        u_not_best = max(u_not)
        # Average contrastive utility
        u_not_avg = sum(u_not) / len(u_not)

        return [u, u_not, u_not_best, u_not_avg]

    #  Compute a locally minimal PAXp (the predicate to study in the next history sub-sequence)
    #  Input: state-action list (state-action list),  state utility (float), predicate (str), scenarios length (int),
    #  proportion threshold (float), importance type (str), importance score method (str),
    #  coefficients to balance between initial and redefined predicate (float list)
    #  Output: partial state, predicate (state)
    def locally_minimal_paxp(self, s_a_list, z, predicate, horizon, delta, imp_type, approach, coefs):
        env_copy, agent_copy = self.preprocess(self.env, self.agent, 'copies', None, self.add_info)
        state = s_a_list[0] if imp_type == 'action' else s_a_list[-1]
        # print('comparison with z: {}'.format(z))

        # modify or not the state structure for several features
        v = self.preprocess(self.env, self.agent, 'pre_locally_minimal_paxp', [state], self.add_info)
        # compute locally minimal paxp
        for i in range(len(v)):
            idx = i if True else len(v) - 1 - i # True/False change the feature evaluation order
            print('try to remove feature {} with value {} --- predicate {}'.format(idx, v[idx], predicate))
            if self.weak_paxp(idx, v, z, predicate, horizon, delta, env_copy, agent_copy, approach, coefs):
                print('Feature {} removed: {}'.format(idx, v[idx]))
                v[idx] = None
        # print('final point v: {}'.format(v))

        # build a state based on v (None replace missing features)
        predicate = self.preprocess(self.env, self.agent, 'post_locally_minimal_paxp', [v], self.add_info)

        print('New predicate: {}'.format(predicate))
        return predicate

    #  Check whether subset v after removing the feature f_i is a weakPAXp or not
    #  Input: removed feature index (int), subset of set features (state), state utility (float), predicate (str),
    #  scenarios length (int), environment (Environment), agent(s) (list) (Agent (list)), importance score method (str),
    #  coefficients to balance between initial and redefined predicate (float list)
    #  Output: (bool)
    def weak_paxp(self, i, v, z, predicate, horizon, delta, env, agent, approach, coefs):
        nb_sample = self.add_info['nb_sample']
        states = self.sample(env, v, i, nb_sample, self.add_info)
        print('len(states) sampling result: {} - feature {}'.format(len(states), i))
        print()

        geq_u = 0 # number of times sample state has a greater than or equal utility than the state under study

        # Compute average utility
        for state in states:
            #print('state: {}'.format(state))
            env, agent, state = self.preprocess(env, agent, 'weak_paxp', state, self.add_info) # only for DC (update map and dead agent attribute)

            # Perform a scenario search if the initial state is not terminal
            if not self.is_terminal(state, env):
                # predict action
                action = self.predict(agent, state)
                # print('agent s action: {}'.format(action))

                # get available actions (for C4 env, actions depend on the state)
                actions = self.get_actions(state, env)

                # compute 'state' utility
                s_a_list = [state, action]
                S, _ = self.first_step(s_a_list, actions, env, agent, 'action', predicate)
                env, agent = self.preprocess(env, agent, 'impScore', s_a_list, self.add_info)
                # print(S)

                S = self.succ(S, horizon, approach, env, agent, 'action', predicate)
                env, agent = self.preprocess(env, agent, 'impScore', s_a_list, self.add_info)

                if self.add_info.get('initial_s'):
                    self.add_info['initial_s'] = s_a_list[0]

                u, _ = self.u(S, [], env, agent, predicate, coefs)

            # Otherwise, directly compute its utility
            else:
                u, _ = self.u([(state, 1.0, 0)], [], env, agent, predicate, coefs)
                # print('utility of state {} for predicate {}: {} '.format(state, predicate, u))

            geq_u += round(u, 5) >= z
            # print('geq_u: {}'.format(geq_u))

        # different samples have been studied for the feature membership
        if len(states):
            print('mean classifier prediction: {}'.format(geq_u / len(states)))
            print()
            return (geq_u / len(states)) >= delta
        # specific case: constant feature. Since it only has one possible value, the feature is not studied (out of PAXp)
        else:
            return True

    # ------------------------ Tool methods --------------------------------

    #  Get first set of states for the succ function (scenarios generation)
    #  Input: state-action list (state-action list), available actions from s (int list), environment (Environment),
    #  agent(s) (list) (Agent (list)), importance type (str), predicate (str)
    #  Output: Set of states by doing pi(s) from s (state-probability list), Set of states by doing all
    #  a' != pi(s) from s (state-probability list list)
    def first_step(self, s_a_list, actions, env, agent, imp_type, predicate):
        S = []
        S_not = []
        S_not_tmp = []
        fixed_horizon = self.add_info['fixed_horizon']

        # action importance
        if imp_type == 'action':
            s, a = s_a_list
            # print('s: {}'.format(s))
            # print('a: {}'.format(a))
            # print('actions:{}'.format(actions))

            for action in actions:
                # specific case: multi-agent setting --> only action of the agent id is replaced
                if self.context == 'multi':
                    a_tmp = deepcopy(a)
                    a_tmp[self.add_info['pred_params'][0] - 1] = action
                    action = a_tmp

                # print('action to test: {} -- original action of the agent: {}'.format(action, a))
                for p, new_s in self.transition(s, action, env, 'none', add_info=self.add_info, imp_type=imp_type):
                    r = 0 if fixed_horizon else self.u([(new_s, 1.0, 0)], [], env, agent, predicate)[1]
                    if action == a:
                        S.append((new_s, p, r))

                    else:
                        S_not_tmp.append((new_s, p, r))

                if action != a:
                    S_not.append(S_not_tmp)
                    S_not_tmp = []

        # transition importance
        else:
            s, a, s_next = s_a_list
            # Limited number of transitions or not in multi-agent setting
            if self.context == 'multi':
                history_tr = self.preprocess(env, None, 'firstStep', s_a_list, self.add_info)
                self.add_info['history_tr'] = history_tr

            for _, new_s in self.transition(s, a, env, 'none', add_info=self.add_info, imp_type=imp_type):
                r = 0 if fixed_horizon else self.u([(new_s, 1.0, 0)], [], env, agent, predicate)[1]
                if new_s != s_next:
                    S_not.append([(new_s, 1.0, r)])
                else:
                    S.append((new_s, 1.0, r))
        return S, S_not

    #  Check whether the state is terminal or not
    #  Input: state (state), environment (Environment)
    #  Output: (bool)
    def is_terminal(self, state, env):
        return self.terminal(state, env, self.add_info)

    #  Store action utilities into a CSV file
    #  Input: file name (str), state action list (state-action list), set of utilities computed from s
    #  (float list), predicate (str), importance score method (str)
    #  Output: None
    def store(self, csv_file, s_a_list, utilities, predicate, approach):
        if self.add_info.get('agent_id'):
            agent_id = self.add_info['agent_id']
        else:
            agent_id = None
        # first line
        self.init_csv(csv_file, approach, predicate)
        if len(utilities) == 4:
            u, u_not, u_not_best, u_not_avg = utilities
        else: # deal with specific importance computation where only one action is available to the agent (e.g. C4)
            u, _ = utilities
            u_not, u_not_best, u_not_avg = 0, 0, 0
        if len(s_a_list) == 2:
            s, a = s_a_list
        else:
            s, a, s_next = s_a_list
        # --------------------------- Display info -------------------------------------
        # action importance
        if len(s_a_list) == 2:
            # mono agent
            if not agent_id:
                explanation_info = "By doing action "+ str(a) +" from state "+ str(s) +", the predicate is respected in "+ str(u * 100) +"% \n By doing the best contrastive action, it is respected in "+ str(u_not_best * 100) +"% \n By not doing the action "+ str(a) +", the average respect of the predicate is "+ str(u_not_avg * 100) +"% \n"
            # multi agent
            else:
                explanation_info = "By doing action "+ str(a) +" from states "+ str(s) +" ("+ str(s[agent_id - 1]) +": state of the agent to explain), the predicate is respected in "+ str(u * 100) +"% \n By doing the best contrastive action, it is respected in "+ str(u_not_best * 100) +"% \n By not doing the action "+ str(a) +", the average respect of the predicate is "+ str(u_not_avg * 100) +"% \n"
        # transition importance
        else:
            # mono agent
            if not agent_id:
                explanation_info = "By doing action "+ str(a) +" from state "+ str(s) +", and arriving in state "+ str(s_next) +", the predicate is respected in "+ str(u * 100) +"% \n By doing the best contrastive transition, it is respected in "+ str(u_not_best * 100) +"% \n By not arriving in state "+ str(s_next) +", the average respect of the predicate is "+ str(u_not_avg * 100) +"% \n"
            # multi agent
            else:
                explanation_info = "By doing action "+ str(a) +" from states "+ str(s) +" ("+ str(s[agent_id - 1]) +": coordinates of the agent to explain), and arriving in states "+ str(s_next) +", the predicate is respected in "+ str(u * 100) +"% \n By doing the best contrastive transition, it is respected in "+ str(u_not_best * 100) +"% \n By not arriving in states "+ str(s_next) +", the average respect of the predicate is "+ str(u_not_avg * 100) +"% \n"
        print(explanation_info)
        #  ------------------------ Store in CSV ----------------------------------------
        with open(csv_file, 'a') as f:
            writer = csv.writer(f)
            # action importance
            if len(s_a_list) == 2:
                # mono agent
                if not agent_id:
                    row1 = ['{}-{}'.format(s, a), 'Action : {}'.format(a), u]
                # multi agent
                else:
                    row1 = ['{}-{}-{}'.format(s, s[agent_id - 1], a), 'Action : {}'.format(a), u]
                row2 = ['', 'Other actions: Best', u_not_best]
                row3 = ['', 'Other actions: Average', u_not, u_not_avg]
            # transition importance
            else:
                # mono agent
                if not agent_id:
                    row1 = ['{}-{}-{}'.format(s, a, s_next), 'Transition : {}'.format(s_next), u]
                # multi agent
                else:
                    row1 = ['{}-{}-{}-{}'.format(s, s[agent_id - 1], a,
                                                 s_next[agent_id - 1]), 'Transition : {}'.format(s_next), u]
                row2 = ['', 'Other transitions: Best', u_not_best]
                row3 = ['', 'Other transitions: Average', u_not, u_not_avg]
            writer.writerow(row1)
            writer.writerow(row2)
            writer.writerow(row3)
            writer.writerow('')  # blank line
        return

    #  Init CSV file
    #  Input: file name (str), importance score method (str), predicate (str)
    #  Output: None
    def init_csv(self, csv_file, approach, predicate):
        pred_params = self.add_info['pred_params']
        if isinstance(predicate, list):
            str_pred = predicate[0] + '_' + predicate[1]
        else:
            str_pred = predicate
        with open(csv_file, 'a') as f:
            writer = csv.writer(f)
            if pred_params:
                writer.writerow([approach.upper(), '', str_pred + ': ' + ' / '.join([str(p) for p in pred_params]), 'Proportion'])
            else:
                writer.writerow([approach.upper(), '', str_pred, 'Proportion'])
        return

    #  Predict the action the agent(s) perform from a specific state using its (their) policy
    #  Input: agent or agents list (Agent (list)) and  state or states list (state (list))
    #  Output: action or actions list (int (list))
    def predict(self, agent, state):
        # mono agent setting
        if self.context == 'mono':
            if 'net' in self.add_info:
                if self.name == 'C4':
                    action = agent.predict(state, net=self.add_info['net'])
                elif self.name == 'DO':
                    action, _ = agent.predict(np.array(state[0]), deterministic=True)
            else:
                if self.name == 'FL':
                    action = agent.predict(state)
                else:
                    action = agent[state]
        # multi-agent setting
        else:
            if self.name == 'DC':
                action = [ag.predict(self.add_info['net'], obs=state[idx]) for idx, ag in enumerate(agent)]
            else:
                action = None
        return action

    #  Evaluate the action pi(s) from a state s in the history through the importance score process
    #  Input: history (state-action list), state-action index (int), environment (Environment), agent(s) list
    #  (Agent list), importance type (str), importance score - utility - state-action tuples computed from the history
    #  (float - float - state-action tuple list), importance score method (str), predicate (str), scenarios length
    #  (int), file name (str), coefficients to balance between initial and redefined predicate (float list)
    #  Output: None
    def evaluate(self, H, i, env, agent, imp_type, data, approach, predicate, k, csv_file, coefs):
        print('horizon: ', k)
        actions = []
        # extract info from the history
        if imp_type == 'action':
            actions = self.get_actions(H[i], env)
            s_a_list = H[i], H[i + 1]
        else:
            s_a_list = H[i], H[i + 1], H[i + 2]

        if not self.constraint(s_a_list[1], imp_type, self.add_info):
            tmp_env, tmp_agent = self.preprocess(env, agent, 'hxp', s_a_list, self.add_info)

            # calculate importance score and return couple importance score, utility of action in history
            score, utility = self.imp_score(s_a_list, k, approach, tmp_env, tmp_agent, actions, imp_type, predicate,
                                   csv_file, coefs)
            # store data
            data.append([score, utility, s_a_list])

    #  Split predicate into name, parameters
    #  Input: predicate name, predicate parameters (str)
    #  Output: name (str), parameters (int / int list)
    def extract(self, predicate):
        if predicate.find(" ") != -1:
            predicate_name = predicate.split(" ")[0]
            str_params = predicate.split(" ")[1:]
            params = []
            int_list = False
            for p in str_params:
                # int list case
                if p[0] == '[':
                    if ',' in p:
                        params.append([int(s) for s in p[1:-1].split(',')])
                    else:
                        int_list = True
                        if p[-1] != ']':
                            params.append(int(p[1:]))
                        else:
                            params.append(int(p[1:-1]))
                elif p[0] == '(':
                    params.append(tuple(int(s) for s in p[1:-1].split(',')))
                # int list case
                elif p[-1] == ']':
                    int_list = True
                    params.append(int(p[:-1]))
                # int case
                else:
                    params.append(int(p))
            if int_list:
                params = [params]
            return predicate_name, params
        else:
            return predicate, None

    #  Check whether the predicate is empty or not
    #  Input: predicate
    #  Output: bool
    def is_none_predicate(self, predicate):
        # 2d matrix
        if self.name in ['DO', 'C4']:
            for i  in range(len(predicate)):
                for j in range(len(predicate[0])):
                    if predicate[i][j] is not None:
                        return False
        # 1d vector
        elif self.name == 'FL':
            for feature in predicate:
                if feature is not None:
                    return False
        # multi-dimension vectors
        elif self.name == 'DC':
            # 1d vector
            for feature in predicate[1]:
                if feature is not None:
                    return False
            # 2d matrix
            for i in range(len(predicate)):
                for j in range(len(predicate[0])):
                    if predicate[i][j] is not None:
                        return False

        return True
