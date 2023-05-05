import numpy as np
import torch
# import random
# import time

from code_Singing_birds import FullRankRNN as rnn
from code_Singing_birds import RDM_task as rdm


class REINFORCE:
    
    def __init__(self, input_size=3, hidden_size=128, output_size=3,
                 deltaT=20., noise_std=0, alpha=0.1, lr=0.7):

        self.actor_network = rnn.FullRankRNN(input_size, hidden_size, output_size,
                                             noise_std=noise_std, alpha=alpha, rho=0.8,
                                             train_wi=True, train_wo=True)
        # TODO: should output a continuous value
        self.critic_network = rnn.FullRankRNN(input_size=hidden_size+1, hidden_size=hidden_size, output_size=1,
                                              noise_std=noise_std, alpha=alpha, rho=0.8,
                                              train_wi=True, train_wo=True)

        self.task = rdm.RandomDotMotion(dt=deltaT)

        # loss

      

    def experience(self, n_trs):

        observations = []
        actions = []
        rewards = []
        probs = torch.unsqueeze(torch.zeros(3), 0)
        log_probs = torch.unsqueeze(torch.zeros(3), 0)
        full_mask = torch.unsqueeze(torch.zeros(3), 0)
        values = []
        gt = []
        n_trs = n_trs
        time_step = 0
        store_trial_begin = [time_step]
        trial_index = 0

        self.task.reset()
        action = 0

        while trial_index < n_trs:           
            
            ob, reward, done, info = self.task.step(action=action)
            observations.append(ob)
            rewards.append(reward)

            ob = torch.Tensor(np.array([ob]))
            ob = torch.unsqueeze(ob, 0)
            
            #name_load = 'FullRankRNN'
            #self.actor_network.load_state_dict(torch.load("../code_eLIFE/models/"+name_load+".pt", map_location='cpu'))
                        
            action_probs, trajs = self.actor_network(ob, return_dynamics=True)
            #print("hey", action_probs)
            
            p = action_probs[0][0].clone().detach().numpy()
            #print(p)
            action = np.random.choice(np.arange(len(p)), p=p) # 0, 1, 2: fix, right, left
            mask = torch.zeros(len(p))
            mask[action] = 1
            full_mask = torch.cat((full_mask, torch.unsqueeze(mask, 0)))
            actions.append(action)
            log_prob = torch.log(action_probs.clone())
            probs = torch.cat((probs, torch.unsqueeze(action_probs[0][0], 0)))
            log_probs = torch.cat((log_probs, torch.unsqueeze(log_prob[0][0], 0)))
            
            action = torch.Tensor([action])
            value = 0
            #value = self.critic_network(torch.cat((action, trajs))) #QUESTO OUTPUT È SBAGLIATISSIMO!! E NON CI VA SOFTMAX IQC!
            values.append(value)
            
            new_trial = info["new_trial"]
            if new_trial:
                trial_index = trial_index + 1
                store_trial_begin.append(time_step+1)
                gt.append(info["gt"])

            time_step = time_step + 1
        
        probs = probs[1:]
        log_probs = log_probs[1:]
        full_mask = full_mask[1:]
        observations = np.asarray(observations)
        return observations, rewards, actions, probs, log_probs, full_mask, values, store_trial_begin, gt
    
    
   
    def lossf(self, log_probs, full_mask):
        
        loss = (full_mask * log_probs)
        
        print("yulu",loss)
        
        loss = loss.sum(dim=-1).sum(dim=-1)
        
        print("huhu",loss)
        
        return loss
    
    
        
    def learning(self, n_trs, lr): 
        
        optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=lr)
        
        #with torch.no_grad():        
        #    observations, rewards, actions, probs, log_probs, values, store_trial_begin, gt = self.experience(n_trs)
        
        cum_rho = []
        tau_r = np.inf  # Song et al. set this to 10s only for reaction time tasks
        gradient_per_trial = 0
                
        optimizer.zero_grad()
        observations, rewards, actions, probs, log_probs, full_mask, values, store_trial_begin, gt = self.experience(n_trs)
        print(log_probs)
        loss = self.lossf(log_probs, full_mask)
        loss.backward()
        """
        for i in range(n_trs):
            
            start = int(store_trial_begin[i])
            stop = int(store_trial_begin[i+1])
            
            trial_rewards = np.array(rewards[start:stop])
            trial_actions = actions[start:stop]
            trial_log_probs = log_probs[start:stop]
            trial_values = values[start:stop]
            cumulative_rewards = [] # np.zeros(len(trial_rewards))

            print(trial_log_probs, "\n\n")
            
            for j in range(len(trial_rewards)):
                
                mask = torch.zeros(len(probs[0]))
                mask[trial_actions[j]] = 1
                
                loss = (mask * trial_log_probs[j]).sum()
                
                disc_rew = [r*np.exp(-(i_r+1)/tau_r) for i_r, r in enumerate(trial_rewards[j+1:])]
                cumulative_rewards.append(np.sum(disc_rew))
                
                #print(trial_log_probs[j])
                loss.backward()
                #trial_log_probs[j].detach_()
                #output.detach_()
                print("done")
                
                #gradient_per_trial += trial_log_probs[j].backward() * (np.sum(disc_rew) - trial_values[j])
                
            #cum_rho.append(cumulative_rewards)
            
        #gradient = gradient / n_trs
        """
        optimizer.step()

    def train(self, num_trial, updating_rate, lr):
        
        n_trs = int(num_trial * updating_rate)
        train_iterations = 1 / updating_rate

        for i in range(train_iterations):
            observations, policies, actions, rewards = self.experience(n_trs, kwarg)
            learning(self, n_trs, policies, rewards, lr)