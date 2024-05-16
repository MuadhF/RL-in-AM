from gym import Env
from gym.spaces import Box, Discrete, Tuple
from helpers import *
import numpy as np
import pandas as pd
import EagarTsai as et
import json

# Reading material properties from a JSON file.
f = open("material_params.json")
data = json.load(f)

# Assigning the values to variables so they can be used in the class below
thickness = data['thickness']
T0 = data['T0']
Tm = data['Tm']
Tb = data['Tb']
alpha_min = data['alpha_min']
rho = data['rho']
cp = data['cp']
ap = data['ap']
sigma = data['sigma']
kappa = data['kappa']
L = data['L']
tens = data['tens']
visc = data['visc']

f.close()


class CustomEnv(Env):
    
    """
    The initialisation takes in two mandatory arguments - parameters and timestep. parameters is a list of the states the agent can
    be in, and timestep is a number representing the total number of timesteps in an episode. An optional argument 'qtable' can be 
    given, which is a Pandas DataFrame containing the Q-values for each state-action pair.
    """
    def __init__(self, parameters, timestep, alpha, gamma, epsilon, qtable = None):
        
        # Initialising the action space using the appropriate helper function
        self.action_space = action_space()
        
        # Creates a space of discrete numbers representing all possible parameter combinations.
        # The space starts from value 0.
        self.observation_space = Discrete(len(parameters))
        
        # Assign a random observation space as the initial state.
        self.state = self.observation_space.sample()
        
        # Assigns the list of state spaces to an attribute called parameters.
        self.parameters = parameters
        
        # Create an empty Q table.
        if qtable is None:
            self.qtable = np.zeros((self.observation_space.n, len(self.action_space)))
        
        # If the user already has given a Q-table to use.
        else:
            self.qtable = qtable
        
        # Initialising various hyperparameters and variables.
        self.timestep = timestep    # Timestep
        self.episode = 0            # Number of episodes
        self.epsilon = epsilon      # Epsilon (for greedy algorithm)
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount rate
        self.steps = 0              # Number of steps taken
        self.rmax = 0               # Individual maximum reward obtained
        self.optimal_state = -1     # Current optimal state (initialised as -1)
        self.optimal_steps = 0      # Number of steps taken to reach the current optimal state
        

        # Initialise a DataFrame for storing desired parameters.
        self.results = pd.DataFrame(columns = ['power', 'speed', 'hatch', 'pi1', 'pi2', 'ved', 'state', 'reward',\
                                               'step', 'episode'])       
        
        # Creating a list to store the rewards obtained for each state.
        self.state_reward = create_state_reward(len(parameters))
    
    

    """
    The step method implements an action taken by the agent. The method chooses an action and then moves the agent from
    the current state to the next state. It then calculates the pi1 and pi2 values, normalises them and then calculates
    the reward function. The Q-table is updated with the new Q-values and the details about this step is recorded into a 
    DataFrame.
    """
    def step(self, power, speed, hatch, test = False, store = False):
        done = False
        self.steps += 1
        old_state = self.state
       
        # As long as the new state is not the same as the old state, the agent chooses an action and implements it.
        while (self.state == old_state):
            
            # Both the action as well as the index of the action set are returned as output, since the index will be
            # used to update the Q-table.
            action, a = choose_action(self, test)
            self.state = take_action(self.state, action, self.parameters, power, speed, hatch)
  #print("State is", self.state)
        
        # Separating each component of the parameter tuple into the three parameters that are being studied.
        parameter = self.parameters[self.state]
        P = parameter[0]
        v = parameter[1] / 1000     # Converting from mm/s to m/s
        h = parameter[2] / 1000     # Converting from mm to m
        
        # Calculating the volumetric energy density. VED is in J/mm3 so we use the units in mm.
        ved = P / (parameter[1] * parameter[2] * thickness * 1000)
        
        # Calculating pi1 and pi2
        pi1, pi2 = et.calc_pi(P, v, h, thickness, T0, Tm, Tb, alpha_min, rho, cp, ap, sigma, kappa, L, tens, visc)

        # Capping the pi1 values to 40 and pi2 values to 2000.
        if pi1 > 40:
            pi1 = 40
        elif pi2 > 2000:
            pi2 = 2000

        # Normalising pi values. 
        pi1_normalised = pi1 / 40
        pi2_normalised = 1 - (pi2 / 2000)     # Because lower pi2 values are desirable, so this inverse is done.

        # Calculating the reward for this action-state pair using multiple sigmoid functions.
        reward = (5 / (1 + np.exp(-6 * (pi1_normalised - 0.8)))) + (5 / (1 + np.exp(-12 * (pi2_normalised - 0.75)))) \
                    - (2 / (1 + np.exp(17 - 0.1 * ved))) - (2 - (2 / (1 + np.exp(5 - 0.1 * ved))))
        
        # In a few cases, the pi values turn out to be nan, so in this case we let the reward be 0.
        if (np.isnan(reward) or reward < 0):
            reward = 0
            
        # Adding a row with metadata into the results DataFrame.
        row = pd.Series([P, v, h, pi1, pi2, ved, self.state, reward, self.steps, self.episode], \
                        index = self.results.columns)
        self.results.loc[len(self.results)] = row
        
        # Adding the state-reward information to the state_reward dictionary.
        self.state_reward[self.state] = reward
        
        # Updating the Q-table with the new Q-values if it is not a test run
        if not test:
            self.qtable[old_state][a] = (1 - self.alpha) * self.qtable[old_state][a] + \
            self.alpha * (reward + self.gamma * np.max(self.qtable[self.state]))
                
        # Reducing the epsilon value with each step.
        if self.epsilon > 0.2:
            self.epsilon -= 0.000015
        
        # Updating the value of the maximum reward, the current optimal state, and the steps taken to reach it.
        if reward > self.rmax:
            self.rmax = reward
            self.optimal_state = self.state
            self.optimal_steps = self.steps       
        
        # Updating the number of timesteps remaining.
        self.timestep -= 1
        
        # If no more timesteps are left, then done will be returned as True, as well as the reward, and the episode will terminate.
        if self.timestep == 0:
            done = True

        return reward, done
    
    
    """
    The reset method is used before commencing a new episode. This will reinitialise the agent to a random state and reset the
    number of timesteps remaining.
    """
    def reset(self, timestep):
        self.timestep = timestep
        self.state = self.observation_space.sample()
        self.episode += 1
       