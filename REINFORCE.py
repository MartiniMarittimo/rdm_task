import numpy as np
import torch
import time

import FullRankRNN as rnn
import RDM_task as rdm



class REINFORCE:
    
    def __init__(self, input_size=3, hidden_size=128, output_size=3,
                 deltaT=20., noise_std=0, alpha=0.1, lr=0.7):

        self.actor_network = rnn.FullRankRNN(input_size, hidden_size, output_size,
                                             noise_std=noise_std, alpha=alpha, rho=0.8,
                                             train_wi=True, train_wo=True)
        
        self.actor_network.actor_critic(actor=True)
        
        # TODO: should output a continuous value
        self.critic_network = rnn.FullRankRNN(input_size=hidden_size+1, hidden_size=hidden_size, output_size=1,
                                              noise_std=noise_std, alpha=alpha, rho=0.8,
                                              train_wi=True, train_wo=True)

        self.task = rdm.RandomDotMotion(dt=deltaT)
        
        
        
    def loss(self, log_probs, actions, cum_rho, n_trs):
        
        new_mask = torch.zeros(log_probs.size())
        
        for i in range(len(actions)):
            action = actions[i]
            new_mask[i][action] = 1
            
        #assert torch.all(torch.eq(new_mask, full_mask))
        
        loss = (new_mask * log_probs)
        loss = loss.sum(dim=-1)
        loss = loss * cum_rho        
        loss = loss.sum(dim=-1) / n_trs
        
        return loss
    
    
    
    def loss_mse(self, output, target):

        loss = (target - output).pow(2).mean(dim=-1)

        return loss

      

    def experience(self, n_trs):
        
        begin = time.time()

        observations = []
        rewards = []
        
        probs = torch.unsqueeze(torch.zeros(3), 0)
        actions = []
        log_probs = torch.unsqueeze(torch.zeros(3), 0)
        
        values = torch.zeros(0) # TODO: da aggiustare assieme al resto
        
        gt = []
        
        n_trs = n_trs
        trial_index = 0
        time_step = 0
        trial_begins = [time_step]

        self.task.reset()
        action = 0

        while trial_index < n_trs:           
            
            ob, rwd, done, info = self.task.step(action=action)
            observations.append(ob)
            rewards.append(rwd)

            ob = torch.Tensor(np.array([ob]))
            ob = torch.unsqueeze(ob, 0) # tensor of size (1,1,3)

            #name_load = 'FullRankRNN'
            #self.actor_network.load_state_dict(torch.load("../code_eLIFE/models/"+name_load+".pt", map_location='cpu'))
            
            action_probs, trajs = self.actor_network(ob, return_dynamics=True) # action_probs: tensor of size (1,1,3)
            
            p = action_probs[0][0].clone().detach().numpy()
            action = np.random.choice(np.arange(len(p)), p=p) # 0, 1, 2: fix, right, left
            actions.append(action)
            action_log_probs = torch.log(action_probs)
            probs = torch.cat((probs, torch.unsqueeze(action_probs[0][0], 0)))
            log_probs = torch.cat((log_probs, torch.unsqueeze(action_log_probs[0][0], 0)))
            
            action = torch.Tensor([action])
            value = self.critic_network(torch.cat((action, trajs[0][0])))
            values = torch.cat((values, value))
            
            if info["new_trial"]:
                trial_index = trial_index + 1
                trial_begins.append(time_step+1)
                gt.append(info["gt"])

            time_step = time_step + 1
                
        observations = np.asarray(observations)
        probs = probs[1:]
        log_probs = log_probs[1:]
        
        return observations, rewards, actions, probs, log_probs, values, trial_begins, gt
    
   
    
    def learning(self, n_trs, lr): 
        
        begin = time.time()
        
        optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=lr)
        
        #with torch.no_grad():        
        
        optimizer.zero_grad()
        
        observations, rewards, actions, probs, log_probs, values, trial_begins, gt = self.experience(n_trs)

        cum_rho = np.zeros(0)
        tau_r = np.inf  # Song et al. set this to 10s only for reaction time tasks
                
        for i in range(n_trs):
            
            start = int(trial_begins[i])
            stop = int(trial_begins[i+1])
            
            trial_rewards = rewards[start:stop]
            trial_cumulative_rewards = []
            
            for j in range(len(trial_rewards)):
                
                disc_rew = [r*np.exp(-(i_r)/tau_r) for i_r, r in enumerate(trial_rewards[j+1:])]
                trial_cumulative_rewards.append(np.sum(disc_rew))
            
            trial_cumulative_rewards = np.array(trial_cumulative_rewards)            
            cum_rho = np.concatenate((cum_rho, trial_cumulative_rewards))
                    
        cum_rho = torch.Tensor(cum_rho)
        
        loss = self.loss(log_probs, actions, cum_rho - values, n_trs)
        loss.backward(retain_graph=True)
        optimizer.step()

        optimizer.zero_grad()
        loss_mse = self.loss_mse(values, cum_rho)
        loss_mse.backward()
        optimizer.step()
        
        print("It took %fs for %i trials" %(time.time()-begin, n_trs))
    
    
    
    def train(self, num_trial, updating_rate, lr):
        
        n_trs = int(num_trial * updating_rate)
        train_iterations = 1 / updating_rate

        for i in range(train_iterations):
            observations, policies, actions, rewards = self.experience(n_trs, kwarg)
            learning(self, n_trs, policies, rewards, lr)