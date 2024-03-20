import numpy as np

#############################################  Model-base methods ######################################################

#  Policy iteration algorithm
#  Input: environment (Environment), maximal number of episodes (int), maximal number of policy evaluation iterations
#  (int), discount factor (float), threshold used to stop evaluation process (float),
#  use of a gymnasium environment (bool)
#  Output: policy (int:int dict), state value dictionary (int:float dict)
def policy_iteration(env, max_episodes, max_value_iter, gamma, theta=-1, gym_use=False):
    # Init policy, value function, transition, action and state list
    states = stateList(env, gym_use)
    T = env.P
    actions = actionList(env, gym_use)
    V = {s: 0.0 for s in states}
    policy = {s: 0 for s in states}

    # Loop on episodes
    for i in range(max_episodes):
        # Policy evaluation (update V)
        V = policy_evaluation(max_value_iter, states, policy, V, T, gamma, theta)
        # Policy improvement (update policy)
        policy, policy_stable = policy_improvement(actions, states, policy, V, T, gamma)
        # If policy is stable, we end the process
        if policy_stable:
            break

    return policy, V

#  Value iteration algorithm
#  Input: environment (Environment), maximal number of iterations (int), discount factor (float), threshold used to stop
#  evaluation process (float), use of a gymnasium environment (bool)
#  Output: policy (int:int dict), state value dictionary (int:float dict)
def value_iteration(env, max_value_iter, gamma, theta, gym_use=False):
    # Init policy, value function, transition, action and state list
    states = stateList(env, gym_use)
    T = env.P
    actions = actionList(env, gym_use)
    V = {s: 0.0 for s in states}
    policy = {s: 0 for s in states}

    # Similar to policy evaluation without the use of the policy
    for j in range(max_value_iter):
        max_difference = 0
        # Compute for each state the expected return following the policy
        for s in states:
            # Extract the best value
            expected_returns = []
            for a in actions:
                expected_returns.append(discounted_expected_return(T[s][a], gamma, V))
            tmp_sum = max(expected_returns)
            # Update max_difference
            max_difference = max(max_difference, abs(tmp_sum - V[s]))
            # Update V
            V[s] = tmp_sum

        if theta >= 0 and max_difference < theta:
            break

    # Only one policy improvement
    policy, _ = policy_improvement(actions, states, policy, V, T, gamma)

    return policy, V

#  Policy evaluation process
#  Input: maximal number of iterations (int), states list (int list), policy dictionary (int:int dict), state value
#  dictionary (int:float dict), transitions dictionary (int:dict(int:tuple) dict), discount factor (float),
#  threshold used to stop evaluation process (float)
#  Output: state value dictionary (int:float dict)
def policy_evaluation(max_value_iter, states, policy, V, T, gamma, theta=-1):
    for j in range(max_value_iter):
        max_difference = 0
        # Compute for each state the expected return following the policy
        for s in states:
            # Best action from s according to policy
            a = policy[s]
            # Compute discounted expected return
            tmp_sum = discounted_expected_return(T[s][a], gamma, V)
            # Update max_difference
            max_difference = max(max_difference, abs(tmp_sum - V[s]))
            # Update V
            V[s] = tmp_sum
            #print(V[s])

        # Policy evaluation terminates if max_difference is smaller than the threshold theta in classic algorithm
        # Policy evaluation terminates at the end of the loop in modified algorithm
        if theta >=0 and max_difference < theta:
            break

    return V

#  Policy improvement process
#  Input: actions list (int list), states list (int list), policy dictionary (int:int dict), state value dictionary
#  (int:float dict), transitions dictionary ([int,int]:tuple dict), discount factor (float)
#  Output: (improved) policy (int:int dict), bool
def policy_improvement(actions, states, policy, V, T, gamma):
    policy_stable = True
    for s in states:
        # Best action from s according to policy
        a_p = policy[s]
        # Best action from according to V
        a_v = 0
        best_expected_return = -1
        for a in actions:
            tmp_sum = discounted_expected_return(T[s][a], gamma, V)
            if tmp_sum > best_expected_return:
                best_expected_return = tmp_sum
                a_v = a
        # Update policy
        policy[s] = a_v
        # Compare both actions
        if a_p != a_v:
            policy_stable = False

    return policy, policy_stable

#  Compute the discounted expected return from a state-action couple
#  Input: list of transitions obtained by doing an action a from a state s (tuple list), discount factor (float),
#  state value dictionary (int:float dict)
#  Output: discounted expected return (float)
def discounted_expected_return(transitions, gamma, V):
    tmp_sum = 0
    for transition in transitions:
        proba, new_s, r, _ = transition
        tmp_sum += proba * (r + gamma * V[new_s])

    return tmp_sum

#################################################  Q Learning ##########################################################

#  Q-learning algorithm
#  Input: environment (Environment), number of training episodes (int), states list (int list), actions list (int list),
#  learning rate (float), discount factor (float)
#  Output: state-action value dictionary ([int,int]:float dict)
def SarsaMax(env, num_episodes, S, A, alpha, gamma):
    # Init Q table, epsilon table
    Q = {s: {a: 0 for a in A} for s in S}
    epsilon_table = epsilonSchedule(num_episodes)
    # Training loop
    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        epsilon = epsilon_table[i]
        while not done:
            # Choose action
            action = epsilon_greedy(Q, state, A, epsilon)
            # Step
            new_state, reward, done, _, _ = env.step(action)
            # Updating rule
            Q[state][action] = Q[state][action] + alpha*(reward + gamma*max(Q[new_state].values()) - Q[state][action])
            # Update current state
            state = env.s
    return Q

#  Choice of an action from a state during the learning of Q
#  Input: state-action value dictionary ([int,int]:float dict), state (int), actions list (int list),
#  exploration rate (float)
#  Output: action (int)
def epsilon_greedy(Q, state, A, epsilon):
    # Exploratory move
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(A)
    # Greedy move
    else:
        action = predict(Q, state)
    return action

#  Compute a range of decreasing exploration rates
#  Input: number of training episodes (int),  minimal exploration rate (float)
#  Output: list of decreasing exploration rates (float list)
def epsilonSchedule(num_episodes, exp_rate_min=0.05):
    x = np.arange(num_episodes) + 1
    exp_rate_decay = exp_rate_min**(1 / num_episodes)
    y = [max((exp_rate_decay**x[i]), exp_rate_min) for i in range(len(x))]
    return y

#  Learn a Q table and deduce a policy
#  Input: environment (Environment), number of training episodes (int), learning rate (float), discount factor (float),
#  use of a gymnasium environment (bool)
#  Output: policy (int:int dict), state-action value dictionary ([int,int]:float dict)
def Qlearning(env, num_episodes, alpha=.2, gamma=.95, gym_use=False):
    S = stateList(env, gym_use)
    A = actionList(env, gym_use)
    # Learn Q function
    Q = SarsaMax(env, num_episodes, S, A, alpha, gamma)
    # Init policy
    policy = {s: 0 for s in S}
    # Create policy based on Q-function
    for s in policy:
        policy[s] = predict(Q, s)
    return policy, Q

#  Predict the action to perform from s according to Q
#  Input: state-action value dictionary ([int,int]:float dict), state (int)
#  Output: action (int)
def predict(Q, s):
    action_values = Q[s].values()
    return np.argmax(list(action_values))

#  Get the list of states
#  Input: environment (Environment), use of a gymnasium environment (bool)
#  Output: states list (int list)
def stateList(env, gym_use):
    if gym_use:
        S = [i for i in range(env.observation_space.n)]
    else:
        S = env.states
    return S

#  Get the list of available actions
#  Input: environment (Environment), use of a gymnasium environment (bool)
#  Output: actions list (int list)
def actionList(env, gym_use):
    if gym_use:
        A = [i for i in range(env.action_space.n)]
    else:
        A = [i for i in range(env.actions)]
    return A
