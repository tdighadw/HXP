The implementation is divided into *7* folders. The code was done under the 3.8 Python version.  Before testing our implementation, it's *necessary* to install packages of requirements.txt using the 
following pip command: 

```bash
pip install -r requirements.txt
```

Then, before running test or training files, the user must be in the problem directory:
```bash
cd 'FrozenLake'
cd 'DroneCoverage'
cd 'Connect4'
cd 'DynamicObstacles'
cd 'Labyrinth'
```

Find below the main commands to use:
```bash
#####  Frozen Lake  ##### 
# Training of an Agent for the 4x4 map with 10,000 episodes. The name of the trained policy is '4x4_test' (not required command). An agent's state is composed of 5 features.
python3 train.py -policy '4x4_test' -features
# Test the default policy trained in a 4x4 map. By default, the user can ask at each timestep, HXP of maximum length 5. An agent's state is composed of 5 features.
python3 test.py -features
#  Test the default policy trained in a 4x4 map. By default, the user can ask at each time-step, B-HXP of length 6 (sub-sequences of length 3 are studied and delta is set to 0.9). An agent's state is composed of 5 features.
python3 test.py -backward 3,0.9 -k 6 -features

#####  Drone Coverage  #####
# Training of the Agents with 40,000 episodes. (not required command)
python3 train.py -model "Test_Models" -log "Test_Logs" -limit 40000
# Test the default learnt policy. By default, the user can ask at each timestep, HXP of maximum length 5.
python3 test.py
#  Test the default learnt policy. By default, the user can ask at each time-step, B-HXP of length 6 (sub-sequences of length 3 are studied, delta is set to 0.9 and only 10 samples are generated to evaluate a feature for the predicate generation process).
python3 test.py -backward 3,0.9 -k 6 -samples 10

#####  Connect 4  ##### 
#  Training of the Agents with 200,000 episodes. (not required command)
python3 train.py -model 'Test_Models' -log 'Test_Logs' -limit 200000 
#  Test the default learnt policy. By default, the user can ask at each time-step, HXP of maximum length 5.
python3 test.py
#  Test the default learnt policy. By default, the user can ask at each time-step, B-HXP of length 6 (sub-sequences of length 3 are studied, delta is set to 0.9 and only 10 samples are generated to evaluate a feature for the predicate generation process).
python3 test.py -backward 3,0.9 -k 6 -samples 10

#####  DynamicObstacles  ##### 
#  Training of the Agents with 200,000 episodes. (not required command)
python3 train.py -model 'Test_Models' -log 'Test_Logs' -limit 200000 -policy 'test'
#  Test the default learnt policy. By default, the user can ask at each time-step, HXP of maximum length 5.
python3 test.py
#  Test the default learnt policy. By default, the user can ask at each time-step, B-HXP of length 6 (sub-sequences of length 3 are studied, delta is set to 0.9 and only 10 samples are generated to evaluate a feature for the predicate generation process).
python3 test.py -backward 3,0.9 -k 6 -samples 10

#####  Labyrinth  ##### 
#  Training of the Agents with 100 episodes. (not required command)
python3 train.py -policy 'test'
#  Test the default learnt policy in the 'corridor' map. By default, the user can ask at each time-step, HXP of maximum length 5.
python3 test.py
```

# Code Structure #


## Frozen Lake (FL) ##

### File Description ###

The Frozen Lake folder is organised as follows:

* **train.py**: parameterized python file which calls training function for Agent instance, and store learnt Q-table into JSON file.


* **test.py**: parameterized python file which loads learnt Q-table and tests it. This file can be use in three ways:
    * A classic sequence loop (named *user mode*) which starts in the initial state of the chosen map. The agent's policy is used and must be provided by the user if the map is not '4x4'. 
      At each time-step except the first one, the user can ask for a (B-)HXP.
    * A specific computation of a (B-)HXP from a given history. In this case, the user must provide at least the *-spec_his* and *-pre* parameters.
    * A generation of *ep* histories with their last state respecting the predicate *pre*. The user must, at least, provide the *-ep*, *-pre* and *-find_histories* parameters.


* **agent.py**: contains the *Agent* class for the RL agent.


* **env.py**: contains a *MyFrozenLake* class: the Frozen Lake environment (inspired by https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py). Two state representations are available: the first one is simply composed of the agent's position while the second one is composed of 5 features: agent position (P), agent's previous position (PP), position of one of the two holes closest to the agent (HP), the Manhattan distance between the agent's initial position and his current position (PD), and the total number of holes on the map (HN). 

* **HXp_tools.py**: set of specific functions used for the (B-)HXP computation.


* **Q-tables folder**: contains all learnt Q-tables. Each file name starts by *Q_*.


* **Utility folder**: contains CSV files in which the value of the actions utility computed during the (B-)HXPs are stored. For an action performed by the agent from a state s in a history, 
its utility, the best contrastive and average contrastive action utility from s is written on the CSV file.


* **Histories folder**: contains all files in which histories are stored. If several histories are stored in a file, the final state of each history respects the same predicate.

By default, running **train.py** starts a training of Agent of 10,000 episodes, on the 4x4 map and **test.py**
runs a classic testing loop on the 4x4 map. To test HXP for other agents and maps, the user must set the parameters *-map* and *-policy*. 

### Examples ###

The followings bash commands are examples of use of **train.py** and **test.py** files.

**Train:**
```bash
# Training of an Agent for the 4x4 map with 10,000 episodes. The name of the trained policy is '4x4_test' (not required command)
python3 train.py -policy '4x4_test'
# Training of an Agent on 10x10 map with 500,000 episodes and save Q-table in JSON file with a name finishing by "10x10_test"
python3 train.py -map "10x10" -policy "10x10_test" -ep 500000
```
**Test:**
```bash
#####  Test in user mode a policy  ##### 

# Test the default policy trained in a 4x4 map. The user can ask at each timestep, HXP of maximum length 5. HXP highlights the most important action and associated state.
python3 test.py
# Test the default learnt policy on a 10x10 map.The user can ask at each timestep, HXP of maximum length 8. HXP highlights the 2 most important actions and associated states. 
python3 test.py -policy 10x10_proba262 -map '10x10' -k 8 -n 2
#  Test the default policy trained in a 4x4 map. By default, the user can ask at each time-step, B-HXP of length 6 (sub-sequences of length 3 are studied and delta is set to 0.9). An agent's state is composed of 5 features.
python3 test.py -backward 3,0.9 -k 6 -features

#####  Test (B-)HXP from a specific history  ##### 

#  Compute an HXP for a length-8 history, in the 10x10 map. The 'region' predicate is studied and the region is the set of the following states:[14, 15, 16, 24, 25, 26]. This HXP highlights the most important action.
python3 test.py -map 10x10 -policy 10x10_proba262 -k 8 -spec_his 'specpart_k8_10x10.txt'  -pre 'specific_part [14,15,16,24,25,26]'
#  Compute an approximate HXP (one last deterministic transition) for a length-10 history, in the 10x10 map. The 'holes' predicate is studied. This approximate HXP highlights the two most important action.
python3 test.py -map 10x10 -policy 10x10_proba262 -strat last_1 -k 10 -spec_his  'holes_k10_10x10.txt' -pre 'holes' -n 2
#  Compute a B-HXP for a length-12 history, in the 8x8 map. The initial studied predicate is 'win'. Sub-sequences of length 4 are studied, delta is set to 0.7 and only 10 samples are generated for predicate generation process.
python3 test.py -map '8x8' -policy 'features_8x8_proba262' -features -spec_his 'win_k12_8x8_features.txt' -backward 4,0.7 -sample 10 -k 12 -pre 'win'

#####  Produce x histories whose last state respects a certain predicate  #####

#  Find 50 length-5 histories whose last state respects the 'win' predicate in the 10x10 map. Histories are stored in the 'win_k5_50hist.csv' file.
python3 test.py -find_histories -ep 50 -pre 'win' -map 10x10 -policy 10x10_proba262 -csv 'win_k5_50hist.csv' -k 5
```

## Drone Coverage (DC) ##

### File Description ###

The Drone Coverage folder is organised as follows: 

* **train.py**: parameterized python file which calls training function for Agent instance, save info into text files and neural network into *dat* files.


* **test.py**: parameterized python file which loads learnt neural network and tests it. This file can be use in three ways:
    * A classic sequence loop (named *user mode*) which starts in an initial configuration. The agents policy is used. 
      At each time-step except the first one, the user can ask for a (B-)HXP.
    * A specific computation of a (B-)HXP from a given history. In this case, the user most provide at least the *-spec_his* and *-pre* parameters
    * A generation of *ep* histories with their last state respecting the predicate *pre*. The user must, at least, provide the *-ep*, *-pre* and *-find_histories* parameters.


* **agent.py**: contains the *Agent* class for the agents.


* **env.py**: contains a DroneCoverageArea class: the Drone Coverage environment.


* **DQN.py**: this file is divided into 3 parts, it's inspired by https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter25/lib/model.py
and https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/02_dqn_pong.py. 
It contains a *DQN* class which is the neural network used to approximate *Q*, an *ExperienceBuffer* class to store agents 
experiences and functions to calculate the DQN's loss.


* **HXp_tools.py**: set of specific functions used for the (B-)HXP computation.


* **Models folder**: contains already learnt policies for agents. 
The names of the produced DQNs show partially the hyperparameter values. In order, values correspond to the timestep 
limit of training, the timestep where epsilon reaches the value of 0.1 (exploration rate), timestep frequency of 
synchronization of the target neural network, the time horizon of an episode and the use of double-Q learning extension.


* **Logs folder**: contains log files from the learning phase of the agents.


* **Utility folder**: contains CSV files in which the value of the actions utility computed during the (B-)HXPs are stored. For an action performed by the agent from a state s in a history, 
its utility, the best contrastive and average contrastive action utility from s is written on the CSV file.


* **Histories folder**: contains all files in which histories are stored. If several histories are stored in a file, the final state of each history respects the same predicate.

By default, running **train.py** starts a training of the agents of 100,000 episodes and **test.py**
runs a classic testing loop where the user can ask for an HXP. To test (B-)HXP from a specific configuration, 
the user must set a text file for the *-spec_his* parameter. Examples of the used convention for representing a history within a text file can be found in the **Histories** folder.  

To study a predicate *d*, we need to describe it using 3 elements: its name, its point of view (the drone being studied) 
and its type (local/global). Each drone is associated with a number: *Blue* (1), *Green* (2), *Red* (3) and *Yellow* (4). 
For example, if the user is interested in the local predicate *perfect cover* for *Red*, he would write 'perfect_cover 3 3'. 
Another example, if the use is interested in the global predicate *region* based on *Blue*'s actions, he would write 'region 1 0'.

### Examples ###

The number of training time-steps must be greater than or equal to 40,000 according to the other default parameters values. 
The followings bash commands are examples of use of **train.py** and **test.py** files.

**Train:**
```bash
#  Train Agents on 10x10 map with a timestep limit of 40000. It saves info into "Test_Logs" folder and neural networks into "Test_Models" folder
python3 train.py -model "Test_Models" -log "Test_Logs" -limit 40000
#  Train Agents on 10x10 map with a timestep limit of 30000. It saves info into "Test_Logs" folder and neural networks into "Test_Models" folder. The transition function is deterministic since there is no wind. The batch size is set to 16 
python3 train.py -model "Test_Models" -log "Test_Logs" -limit 30000 -no_w -batch 16
```
**Test:**
```bash
#####  Test in user mode a policy  #####

#  Test the default learnt policy on a 10x10 map with 4 agents. Agents start at random positions. The user can ask at each time-step, HXP of maximum length 5. HXP the most important action and associated state.
python3 test.py
#  Test the default learnt policy on a 10x10 map with 4 agents. Agents start at random positions. The user can ask at each timestep, HXP of maximum length 4. HXP highlights the 2 most important actions and associated states.
python3 test.py -k 4 -n 2
#  Test the default learnt policy. By default, the user can ask at each time-step, B-HXP of length 6 (sub-sequences of length 3 are studied, delta is set to 0.9 and only 10 samples are generated to evaluate a feature for the predicate generation process).
python3 test.py -backward 3,0.9 -k 6 -samples 10

#####  Test (B-)HXP from a specific history  ##### 

#  Compute an HXP for a given length-5 history, based on the actions of the Blue drone (Blue id = 1). The 'global region' predicate is studied. This HXP highlights the most important action.
python3 test.py -spec_his All_regions_k5_Ag1.txt -pre 'region 1 0'
#  Compute an approximate HXP (one last deterministic transition) for a length-5 history, based on the actions of the Green drone (Green id = 2). The 'local prefect cover' predicate is studied. This approximate HXP highlights the two most important action.
python3 test.py -spec_his H0L5P_perfect\ coverAg2.txt -strat last_1 -pre 'perfect_cover 2 2' -n 2
#  Compute a B-HXP for a length-10 history, based on Blue's actions. The initial 'local perfect cover' predicate is studied. Sub-sequences of length 3 are studied, delta is set to 1.0 and only 10 samples are generated for predicate generation process.
python3 test.py -spec_his 'perfect_cover_k12.txt' -pre 'perfect_cover 1 1' -backward 3,1.0 -sample 10 -k 12

#####  Produce x histories whose last state respects a certain predicate  #####

# Find 30 length-4 histories whose last state of the Blue drone respects the local perfect cover predicate. Histories are stored in the 'localmaxcover_k6_100hist.csv' file.
python3 test.py -find_histories -ep 30 -pre 'perfect_cover 1 1' -csv 'localperfectcover_k4_30hist.csv' -k 4
```

## Connect4 (C4) ##

### File Description ###

The Connect4 folder is organised as follows:

* **train.py**: parameterized python file which calls training function for Agent instance, save info into text files and neural network into *dat* files.
                Self-play methods (inspired by: https://github.com/Unity-Technologies/ml-agents/tree/release_20_docs) are used for the learning of both Players behavior.   


* **test.py**: parameterized python file which loads learnt neural network and tests it. This file can be use in three ways:
    * A classic sequence loop (named *user mode*) which starts with an empty board. The agent's policy is used for Player 1 and Player 2. To avoid similar plays over the episodes, the Player 2 has 30% of probability to play randomly.  
      The user can ask for a (B-)HXP.
    * A specific computation of an (approximate) HXP from a given history. In this case, the user most provide at least the *-spec_his* and *-pre* parameters
    * A generation of *ep* histories with their last state respecting the predicate *pre*. The user must, at least, provide the *-ep*, *-pre* and *-find_histories* parameters.


* **agent.py**: contains the *Agent* class for the agents.


* **env.py**: contains a class *Connect4*: the Connect4 environment.


* **DQN.py**: this file is divided into 3 parts, it's inspired by https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter25/lib/model.py, 
https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/02_dqn_pong.py and https://codebox.net/pages/connect4. 
It contains a *DQN* class which is the neural network used to approximate *Q*, an *ExperienceBuffer* class to store agents 
experiences and functions to calculate the DQN's loss.


* **HXp_tools.py**: set of specific functions used for the (B-)HXP computation.


* **Models folder**: contains already learnt policies for agents.


* **Logs folder**: contains log files from the learning phase of the agents.


* **Utility folder**: contains CSV files in which the value of the actions utility computed during the (B-)HXPs are stored. For an action performed by the agent from a state s in a history, 
its utility, the best contrastive and average contrastive action utility from s is written on the CSV file.


* **Histories folder**: contains all files in which histories are stored. If several histories are stored in a file, the final state of each history respects the same predicate.

By default, running **train.py** starts a training of the agents of 1,000,000 episodes and **test.py**
runs a classic testing loop where the user can ask for an HXP. To test (B-)HXP from a specific configuration, 
the user must set a text file for the *-spec_his* parameter. Examples of the used convention for representing a history within a text file can be found in the **Histories** folder.  


### Examples ###

The number of training time-steps must be greater than or equal to 200,000 according to the other default parameters values. 
The followings bash commands are examples of use of **train.py** and **test.py** files.

**Train:**
```bash
#  Train Agents with a time-step limit of 1,000,000.
python3 train.py -model 'Test_Models' -log 'Test_Logs' -limit 1000000
#  Train Agents with a timestep limit of 100,000. Only the last two models of the opponent are saved. The learning agent play against these models. Each 50,000 time-steps, the agent who learns change (This is done to alternate between Player 1 and Player 2 learning phase). 
python3 train.py -model 'Test_Models' -log 'Test_Logs' -limit 100000 -window 2 -player_change 50000
```
**Test:**
```bash
#####  Test in user mode a policy  #####

#  Test the default learnt policy. The user can ask at each time-step, HXP of maximum length 5. HXP the most important action and associated state.
python3 test.py
#  Test the default learnt policy. The user can ask at each timestep, HXP of maximum length 6. HXP highlights the 2 most important actions and associated states.
python3 test.py -k 6 -n 2
#  Test the default learnt policy. By default, the user can ask at each time-step, B-HXP of length 6 (sub-sequences of length 3 are studied, delta is set to 0.9 and only 10 samples are generated to evaluate a feature for the predicate generation process).
python3 test.py -backward 3,0.9 -k 6 -samples 10

#####  Test (B-)HXP from a specific history  ##### 

#  Compute an approximate HXP (three last deterministic transitions) for a given length-6 history. The 'control middle column' predicate is studied. This approximate HXP highlights the two most important action.
python3 test.py -spec_his  '6_2_win_hist.txt'  -strat last_3 -k 6 -pre 'control_midcolumn' -n 2
#  Compute an approximate HXP (two last deterministic transitions) for a length-5 history. The 'lose' predicate is studied. This approximate HXP highlights the most important action.
python3 test.py -spec_his  'H0L5P_lose_exp.txt' -strat last_2 -pre 'lose'
#  Compute a B-HXP for a length-12 history. The initial 'win' predicate is studied. Sub-sequences of length 3 are studied, delta is set to 0.8 and only 10 samples are generated for predicate generation process.
python3  test.py -spec_his '12_1_win_hist.txt' -pre win -backward 3,0.8 -sample 10 -k 12

#####  Produce x histories whose last state respects a certain predicate  #####

#  Find 100 length-5 histories whose last configuration respects the '3 in a row' predicate. Histories are stored in the '3inarow_k5_100hist.csv' file.
python3 test.py -find_histories -ep 100 -pre '3inarow' -csv '3inarow_k5_100hist.csv' -k 5
```

## Dynamic Obstacles (DO) ##

### File Description ###

The Dynamic Obstacles folder is organised as follows:

* **train.py**: parameterized python file which calls training function (use of stable-baseline3 library), save info into CSV files and neural network into *zip* files.

* **test.py**: parameterized python file which loads learnt neural network and tests it. This file can be use in three ways:
    * A classic sequence loop (named *user mode*). The user can ask for a (B-)HXP.
    * A specific computation of an (approximate) HXP from a given history. In this case, the user most provide at least the *-spec_his* and *-pre* parameters
    * A generation of *ep* histories with their last state respecting the predicate *pre*. The user must, at least, provide the *-ep*, *-pre* and *-find_histories* parameters.

* **HXp_tools.py**: set of specific functions used for the (B-)HXP computation.


* **Models folder**: contains already learnt policies for agents.


* **Logs folder**: contains log files from the learning phase of the agents.


* **Utility folder**: contains CSV files in which the value of the actions utility computed during the (B-)HXPs are stored. For an action performed by the agent from a state s in a history, 
its utility, the best contrastive and average contrastive action utility from s is written on the CSV file.


* **Histories folder**: contains all files in which histories are stored. If several histories are stored in a file, the final state of each history respects the same predicate.


### Examples ###

**Train:**
```bash
#  Train the agent with a time-step limit of 1,000,000.
python3 train.py -policy 'test' -limit 1000000
#  Train the agent with a time-step limit of 500,000, a batch size set to 16 and a discount factor set to 0.95.
python3 train.py -policy 'test' -limit 500000 -batch 16 -df 0.95
```

**Test:**
```bash
#####  Test in user mode a policy  #####

#  Test the default learnt policy. By default, the user can ask at each time-step, HXP of maximum length 5.
python3 test.py
#  Test the default learnt policy. By default, the user can ask at each time-step, approximate HXP of maximum length 6. The two most important actions are highlighted to the user.
python3 test.py -strat last_3 -k 6 -n 2
#  Test the default learnt policy. By default, the user can ask at each time-step, B-HXP of length 6 (sub-sequences of length 3 are studied, delta is set to 0.9 and only 10 samples are generated to evaluate a feature for the predicate generation process).
python3 test.py -backward 3,0.9 -k 6 -samples 10

#####  Test (B-)HXP from a specific history  ##### 

#  Compute an approximate HXP (two last deterministic transitions) for a given length-5 history. The 'success' predicate is studied.
python3 test.py -spec_his success1_k5.txt -strat last_2  -k 5 -pre success
#  Compute a B-HXP for a length-12 history. The initial 'success' predicate is studied. Sub-sequences of length 3 are studied, delta is set to 0.9 and only 10 samples are generated for predicate generation process.
python3 test.py -spec_his success1_k12.txt -backward 3,0.9 -samples 10  -k 12 -pre success

#####  Produce x histories whose last state respects a certain predicate  #####

#  Find 100 length-15 histories whose last configuration respects the 'success' predicate. Histories are stored in the 'success_k15_100hist.csv' file. 
python3 test.py -find_histories -ep 100 -csv 'success_k15_100hist.csv' -pre success -k 15
````

## Labyrinth (L) ##

### File Description ###

The Labyrinth folder is organised as follows:

* **train.py**: parameterized python file which calls training function (use of one algorithm from *algorithms.py*), save info into CSV files and neural network into *txt* files.


* **test.py**: parameterized python file which loads learnt neural network and tests it. This file can be use in two ways:
    * A classic sequence loop (named *user mode*). The user can ask for a (B-)HXP.
    * A specific computation of an (approximate) HXP from a given history. In this case, the user most provide at least the *-spec_his* and *-pre* parameters

* **algorithms.py**: set of classic RL algorithms such as Policy Iteration, Value Iteration and SarsaMax.


* **env.py**: contains a class *Labyrinth*: the Labyrinth environment.


* **HXp_tools.py**: set of specific functions used for the (B-)HXP computation.


* **Policies folder**: contains already learnt policies for agents.


* **Utility folder**: contains CSV files in which the value of the actions utility computed during the (B-)HXPs are stored. For an action performed by the agent from a state s in a history, 
its utility, the best contrastive and average contrastive action utility from s is written on the CSV file.


* **Histories folder**: contains all files in which histories are stored. If several histories are stored in a file, the final state of each history respects the same predicate.


### Examples ###

**Train:**
```bash
#  Train the agent in the corridor map with 100 episodes
python3 train.py
#  Train the agent in the crossroad map with 200 episodes. The policy is stored in the 'policy_crossroad_det_env'.
python3 train.py -map crossroad -ep 200 -policy 'crossroad_det_env'
```

**Test:**
```bash
#####  Test in user mode a policy  #####

#  Test the default learnt policy in the corridor map. By default, the user can ask at each time-step, HXP of maximum length 5.
python3 test.py
#  Test the default learnt policy in the crossroad map. By default, the user can ask an HXP of length 6.
python3 test.py -map crossroad -policy 'crossroad_det_env' -k 6

#####  Test HXP from a specific history  ##### 

#  Compute an HXP for a given length-8 history in the corridor map. The 'reach exit' predicate is studied.
python3 test.py -spec_his corridor_k8 -k 8 -pre reach_exit
#  Compute an HXP for a given length-5 history in the crossroad map. The 'reach exit' predicate is studied.
python3 test.py -map crossroad -policy 'crossroad_det_env' -spec_his crossroad_k5 -k 5 -pre reach_exit
````




## Additional files / folder ##

The following files and folder are located at the root of the project and are used for the computation of HXPs/B-HXPs, similarity scores, rate of similar most important action selected and run times:

* **HXP.py**: Given an RL problem, this file allows to perform HXP and B-HXP for a given predicate and history (B-HXP is not yet implemented for Labyrinth environment). 


* **similarity.py**: computes HXP and approximate HXPs for each history in a CSV file. In addition to action importance scores, the similarity scores and run-times for each approach is computed then stored for each history. Average and standard deviation of similarity scores and run-times are also computed.  
                     Find below some examples of this file use:
```bash
# Compute action importance scores, similarity scores and run-times for each history in the 'goal_k5_50hist.csv' file. This is done for the Frozen Lake problem, with the default learnt policy (10x10_probas262) in the 10x10 map. The output file is located in the Similarity folder.
python3 similarity.py -file '/FrozenLake/Histories/10x10/50-histories/goal_k5_50hist.csv' -pre goal -problem FL -new_file 'Similarity/FrozenLake/10x10/goal_k5_50hist.csv' -policy 10x10_probas262
# Compute action importance scores, similarity scores and run-times for each history in the 'localperfectcover_k4_30hist.csv' file. This is done for the Drone Coverage problem, with the default learnt policy (tl1600000e750000s50000th22ddqnTrue) in the 10x10 map. The output file is located in the Similarity folder.
python3 similarity.py -file '/DroneCoverage/Histories/30-histories/localperfectcover_k4_30hist.csv' -pre 'one perfect cover' -id 1 -problem DC -new_file 'Similarity/DroneCoverage/10x10/localperfectcover_k4_30hist.csv' -policy '/DroneCoverage/Models/Agent/tl1600000e750000s50000th22ddqnTrue-best_11.69.dat'
# Compute action importance scores, similarity scores and run-times for each history in the '3inarow_k5_100hist.csv' file. This is done for the Connect4 problem, with the default learnt policy (bestPlayerP1_98_P2_96). The output file is located in the Similarity folder.
python3 similarity.py -file '/Connect4/Histories/100-histories/3inarow_k5_100hist.csv' -pre '3inarow' -problem C4 -new_file 'Similarity/Connect4/3inarow_k5_100hist.csv' -policy '/Connect4/Models/bestPlayerP1_98_P2_96.dat'
```

* **same_impaction_rate.py**: Given a CSV file obtained after the use of the *similarity.py* file, this file computes the rate of same most important action returned by the HXP and approximate HXPs.
                    Find below some examples of this file use:
```bash
# Compute, for histories from the Frozen Lake problem, the rate of same most important action returned by the HXP and approximate HXPs
python3 same_impaction_rate.py -file '/Similarity/FrozenLake/10x10/goal_k5_50hist.csv'
# Compute, for histories from the Drone Coverage problem, the rate of same most important action returned by the HXP and approximate HXPs
python3 same_impaction_rate.py -file '/Similarity/DroneCoverage/10x10/localperfectcover_k4_30hist.csv'
# Compute, for histories from the Connect4 problem, the rate of same most important action returned by the HXP and approximate HXPs
python3 same_impaction_rate.py -file '/Similarity/Connect4/3inarow_k5_100hist.csv'
``` 
* **param_study_B-HXP.py**: File used to compare, based on a set of histories, the B-HXP returned as a function of 
                            different values for a given parameter. This file allows a parametric study to be made of 3 
                            parameters related to the B-HXP computation: the length of the sub-sequences studied (*l*), 
                            delta (*d*) and the number of samples used for the evaluation of features in the predicate 
                            generation process (*s*). A call to *param_study_B-HXP.py* is used to study a parameter. The 
                            results are stored in a CSV file. The B-HXPs produced are compared according to 3 metrics: 
                            runtime, action sparsity (i.e. the number of actions returned) and the sparsity of the 
                            predicates generated (i.e. the number of features describing the predicate).

```bash
#  Compute a parameter study for the parameter l for the DC problem. The study is performed over 1,000 length-12 histories. The initial studied predicate is 'max reward' based on Blue's actions. 
python3 parameter_study_B-HXP.py -policy '/DroneCoverage/Models/Agent/tl1600000e750000s50000th22ddqnTrue-best_11.69.dat' -param l -values '[3, 4, 5]' -file '/DroneCoverage/Histories/1000-histories/max_reward_k12_1000hist.csv' -new_file '/param_study_B-HXP/DroneCoverage/max_reward_k12_1000hist_l.csv' -problem DC -pre 'max_reward 1 1' -backward '1, 0.9' -k 12
#  Compute a parameter study for the parameter d for the FL problem in the 8x8 map. The study is performed over 100 length-12 histories. The initial studied predicate is 'holes'. 
python3 parameter_study_B-HXP.py -map '8x8' -policy 'features_8x8_proba262' -features -param delta -values '[0.7, 0.8, 1.0]' -file 'FrozenLake/Histories/100-histories-feats-8x8/feats_holes_k12_100hist.csv' -new_file 'param_study_B-HXP/FrozenLake/feats_holes_k12_100hist_d.csv' -problem FL -pre holes -backward '4, 0.0' -k 12 -samples 10
```


* **Similarity folder**: For each problem, this folder contains all the results obtained with the *similarity.py* file, i.e. files containing for each studied history, action importance scores of different approaches, similarity scores and run-times.


* **param_study_B-HXP folder**: For each problem, this folder contains the study results files for the different parameters *l*, *d* and *s*.

## Remarks ##

If the user wants to compute similarity and the rate of same important action on new histories, he must run successively the *test.py* of a problem, the *similarity.py* and *same_impaction_rate.py* with appropriate parameters.
As a technical detail, for the DC and C4 problem, if CUDA is available, the training and use of DQN will be with the GPU, otherwise with the CPU.