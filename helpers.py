import random
import numpy as np
import pandas as pd



"""
This function creates an array of tuples, where each tuple is an action containing 3 elements. Each element in an action tuple
corresponds to the change to be made on the power, scan speed, and hatch spacing, respectively. Each element can take values
of 0, 1, and 2.

0: Decrease in the process parameter value.
1: No change in the process parameter value.
2: Increase in the process parameter value.
"""
def action_space():
    a = []
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                if (i==1 & j==1 & k==1):     # All three paramater values cannot remain the same, so (1,1,1) is not allowed.
                    pass
                else:
                    a.append((i,j,k))
    return a



"""
This function chooses an action from the action space. It accepts an optional argument test, which indicates if the RL algorithm
is on a train or test run. If it is on a test run, then it will choose the action that corresponds to the highest Q-value in the 
Q-table. If it is on a train run, it will either choose the best action or a random action, depending on the current value of 
the epsilon hyperparameter.

The function returns both the action tuple, as well as the index of the action in the action space.
"""
def choose_action(env, test = False):
    if test:
        # If the maximum Q-value is 0, then a random action will be chosen.
        if np.max(env.qtable[env.state]) == 0:
  #print("Choosing random action")
            # Let a be the index of the Q table column with the highest Q value for the current state.
            a = random.randint(0,len(env.action_space) - 1)
        else:
            a = np.argmax(env.qtable[env.state])
  #print("Choosing best action", a)
    else:
        # Deciding if to choose random action or based on Q table.
        if random.random() < env.epsilon:
            a = random.randint(0,len(env.action_space) - 1)
        else:
            if np.max(env.qtable[env.state]) == 0:
                a = random.randint(0,len(env.action_space) - 1)
            else:
                a = np.argmax(env.qtable[env.state])

    # The chosen action is be the a-th index of the action space.
    return env.action_space[a], a



"""
This function implements the chosen action and transitions the agent from the old state to the new state. It takes as input 
parameters the current state, the action to be taken, the state space list, and each process parameter value list.

The function goes through each element in the action tuple, calculates which state would the agent be in if only that parameter
is updated, and then moves to the next element in the tuple. Once the three elements have been covered, then the state variable
would be the resultant state after taking the action, which is returned.

Considerations are made in the function to ensure the parameter values do not go beyond the predefined range of values.
"""
def take_action(state, action, parameters, power, speed, hatch):
    # If first parameter's action is 0 (decrease) and can be decreased, then decrease it by 1 unit, by moving through the state space.
    if action[0] == 0 and parameters[state][0] != np.min(power):
        state = state - (len(speed) * len(hatch)) 
    # If first parameter's action is 1 (stay the same), then pass.
    elif action[0] == 1:
        pass
    # If first parameter's action is 2 (increase) and can be increased, then increase it by 1 unit, by moving through the state space.
    elif action[0] == 2 and parameters[state][0] != np.max(power):
        state = state + (len(speed) * len(hatch))

    if action[1] == 0 and parameters[state][1] != np.min(speed):
        state = state - len(hatch)
    elif action[1] == 1:
        pass
    elif action[1] == 2 and parameters[state][1] != np.max(speed):
        state = state + len(hatch)
    
    if action[2] == 0 and parameters[state][2] != np.min(hatch):
        state = state - 1
    elif action[2] == 1:
        pass
    elif action[2] == 2 and parameters[state][2] != np.max(hatch):
        state = state + 1
    
    return state



"""
This function creates a dictionary of states where each has an initial reward of 0, which will be updated as the agent
explores the environment. It takes as input the number of state spaces and returns the initialised dictionary.
"""
def create_state_reward(n_states):
    state_reward = {}
    for i in range(n_states):
        state_reward[i] = 0
    
    return state_reward