import numpy as np
import torch
import neurogym as ngym

import random
import time

from files import FullRankRNN as rnn 
from files import RDM_task as rdm 


class REINFORCE:
    
    def __init__(self, input_size=3, hidden_size=128, output_size=3,
                 num_trial=1000, updating_rate=0.05):
        
        self.A_network = rnn.FullRankRNN(input_size, hidden_size, output_size, noise_std, alpha, rho = 0.8,
                                   train_wi = True, train_wo = True)
        
        self.C_network = rnn.FullRankRNN(input_size, hidden_size, output_size, noise_std, alpha, rho = 0.8,
                                   train_wi = True, train_wo = True)
        
        self.task = rdm.RandomDotMotion(deltaT=20.):

        #loss
    
    
    def action_selection(network_policy): #network_policy: fix, right, left
        
        action = 0 #fixation
        
        pfix = np.random.uniform()
        
        if pfix > network_policy[0]:
            pright = np.random.uniform()
            renorm = network_policy[1] / (network_policy[1] + network_policy[2])
            
            if pright < renorm:
                action = 1 #right
            else:
                action = -1 #left
        
        return action
    
    
    def data_collection(self, num_trial, updating_rate):
        
        batch_size = int(num_trial*updating_rate)
        trial_index = 0
        observations = []
        policies = []
        actions = []
        rewards = []
        
        while trial_index != batch_size:
            trial = self.task._new_trial(**kwarg)
            trial_index += 1
            next_step = True
            while next_step == True:
                ob_now, reward, bool, dic = self.task._step(action)
                next_step = dic['next_step']
                network_policy = self.A_network(ob_now)
                action = action_selection(network_policy)

                if action == 0:
                    policies.append(network_policy[0])
                elif action == 1:
                    policies.append(network_policy[1])
                elif action == -1:
                    policies.append(network_policy[2])
                observations.append(ob_now)
                policies.append(policy)
                rewards.append(reward)
                actions.append(action)
                
        return observations, policies, actions, rewards
    
    def train(self, num_trial, updating_rate):
        
        train_iterations = 1 / updating_rate
        
        for i in range(train_iterations):
            observations, policies, actions, rewards = self.data_collection(num_trial, updating_rate)
            gradient = 


