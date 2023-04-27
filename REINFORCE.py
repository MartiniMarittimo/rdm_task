import numpy as np
# import torch
# import neurogym as ngym

# import random
# import time

from files import FullRankRNN as rnn
from files import RDM_task as rdm


class REINFORCE:
    def __init__(self, input_size=3, hidden_size=128, output_size=3,
                 num_trial=1000, updating_rate=0.05, noise_std=0, alpha=0.1):

        self.actor_network = rnn.FullRankRNN(input_size, hidden_size, output_size,
                                             noise_std, alpha, rho=0.8,
                                             train_wi=True, train_wo=True)
        # TODO: should output a continuous value
        self.critic_network = rnn.FullRankRNN(input_size, hidden_size, output_size,
                                              noise_std, alpha, rho=0.8,
                                              train_wi=True, train_wo=True)

        self.task = rdm.RandomDotMotion(deltaT=20.)

        # loss

    def action_selection(action_probs):  # network_policy: fix, right, left
        return np.random.choice(np.arange(len(action_probs)), p=action_probs)

    def experience(self, n_trs, **kwarg):

        trial_index = 0
        observations = []
        policies = []
        actions = []
        rewards = []

        while trial_index <= n_trs:
            # trial = self.task._new_trial(**kwarg)
            trial_index += 1
            next_step = True
            action = 0
            while next_step:
                ob_now, reward, bool, dic = self.task.step(action)
                next_step = dic['next_step']
                action_probs = self.actor_network(ob_now)
                action = self.action_selection(action_probs=action_probs)
                policies.append(action_probs[action])
                observations.append(ob_now)
                rewards.append(reward)
                actions.append(action)

        return observations, policies, actions, rewards

    def train(self, num_trial, updating_rate):

        train_iterations = 1 / updating_rate

        for i in range(train_iterations):
            observations, policies, actions, rewards = self.experience(
                num_trial, updating_rate)
            # gradient = 0
