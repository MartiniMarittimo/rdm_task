import numpy as np
import torch

# import random
# import time

from files import FullRankRNN as rnn
from files import RDM_task as rdm


class REINFORCE:
    def __init__(self, input_size=3, hidden_size=128, output_size=3,
                 num_trial=1000, updating_rate=0.05, noise_std=0, alpha=0.1, lr=0.7):

        self.actor_network = rnn.FullRankRNN(input_size, hidden_size, output_size,
                                             noise_std, alpha, rho=0.8,
                                             train_wi=True, train_wo=True)
        # TODO: should output a continuous value
        self.critic_network = rnn.FullRankRNN(input_size, hidden_size, output_size,
                                              noise_std, alpha, rho=0.8,
                                              train_wi=True, train_wo=True)

        self.task = rdm.RandomDotMotion(deltaT=20.)

        # loss

    def action_selection(action_probs):
        return np.random.choice(np.arange(len(action_probs)), p=action_probs) # 0, 1, 2: fix, right, left

    def experience(self, n_trs, **kwarg):

        trial_index = 0
        observations = []
        policies = []
        actions = []
        rewards = []

        while trial_index <= n_trs:
            # trial = self.task._new_trial(**kwarg)
            trial_index += 1
            new_trial = True
            action = 0
            while not new_trial:
                ob, reward, done, info = self.task.step(action)
                new_trial = info['new_trial']
                action_probs = self.actor_network(ob)
                action = self.action_selection(action_probs=action_probs)
                
                policies.append(action_probs[action])
                observations.append(ob)
                rewards.append(reward)
                actions.append(action)

        policies = torch.Tensor(policies)  
        
        return observations, policies, actions, rewards
    
    def learning(self, n_trs, policies, rewards, lr): 
        
        optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=lr)

        policies = np.log(policies)
        gradient = 0
        cumulative_reward = []
        
        for n_and_t in range(len(policies)):
            gradient += policies[n_and_t].backward() * cumulative_reward[n_and_t]
            
        gradeint = gradient / n_trs
        

    def train(self, num_trial, updating_rate, lr):
        
        n_trs = int(num_trial * updating_rate)
        train_iterations = 1 / updating_rate

        for i in range(train_iterations):
            observations, policies, actions, rewards = self.experience(n_trs, kwarg)
            learning(self, n_trs, policies, rewards, lr)