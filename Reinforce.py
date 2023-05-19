import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import time

import FullRankRNN as rnn
import RDMtask as rdm



class REINFORCE:
    
    def __init__(self, input_size=3, hidden_size=128, output_size=3,
                 deltaT=20., noise_std=0, alpha=0.2, name_load=None,
                 train_wi_a=True, train_wrec_a=True, train_wo_a=True,
                 train_wi_c=True, train_wrec_c=True, train_wo_c=True):        
        
        self.actor_network = rnn.FullRankRNN(input_size, hidden_size, output_size,
                                             noise_std=noise_std, alpha=alpha, rho=0.8,
                                             train_wi=train_wi_a, train_wrec=train_wrec_a, train_wo=train_wo_a)
        
        if name_load is not None:
            self.actor_network.load_state_dict(torch.load(name_load, map_location='cpu'))
        
        self.actor_network.actor_critic(actor=True)
        
        self.critic_network = rnn.FullRankRNN(input_size=hidden_size+1, hidden_size=hidden_size, output_size=1,
                                              noise_std=noise_std, alpha=alpha, rho=0.8,
                                              train_wi=train_wi_c, train_wrec=train_wrec_c, train_wo=train_wo_c)

        self.task = rdm.RandomDotMotion(dt=deltaT)
        
        self.hidden_size = hidden_size
        
        
    def loss(self, log_probs, actions, cum_rho, n_trs):
        
        new_mask = torch.zeros(log_probs.size())
        
        for i in range(len(actions)):
            action = actions[i]
            new_mask[i][action] = 1
            
        #assert torch.all(torch.eq(new_mask, full_mask))
        
        loss = (new_mask * log_probs)
        loss = loss.sum(dim=-1)
        loss = loss * cum_rho        
        loss = loss.sum(dim=-1) / (-n_trs)
        
        return loss
    
    
    
    def loss_mse(self, output, target):

        loss = (target - output).pow(2).mean(dim=-1)

        return loss

      

    def experience(self, n_trs):
        
        begin = time.time()

        observations = []
        rewards = []
        
        #probs = torch.unsqueeze(torch.zeros(3), 0)
        log_probs = torch.unsqueeze(torch.zeros(3), 0)
        actions = []
        
        values = torch.zeros(0) 
        
        gt = []
        errors = []
        
        n_trs = n_trs
        trial_index = 0
        time_step = 0
        trial_begins = [time_step]

        self.task.reset()
        action = 0
        
        h0_actor = torch.zeros(self.hidden_size)
        h0_critic = torch.zeros(self.hidden_size)

        while trial_index < n_trs:           
            
            ob, rwd, done, info = self.task.step(action=action)
            observations.append(ob)
            rewards.append(rwd)
            #print("ob+rwd", ob, rwd)
            
            ob = torch.Tensor(np.array([ob]))
            ob = torch.unsqueeze(ob, 0) # tensor of size (1,1,3)
                
            action_probs, trajs = self.actor_network(ob, return_dynamics=True, h0=h0_actor) 
            #print("\naction_probs+trajs", action_probs, trajs)
            
            p = action_probs[0][0].clone().detach().numpy()
            action = np.random.choice(np.arange(len(p)), p=p) # 0, 1, 2: fix, right, left
            actions.append(action)
            action_t = torch.Tensor([action])
            relu_trajs = self.critic_network.non_linearity(trajs[0][0])
            in_for_critic = torch.unsqueeze(torch.unsqueeze(torch.cat((action_t, relu_trajs.detach())),0),0)
            log_action_probs = torch.log(action_probs)
            log_probs = torch.cat((log_probs, torch.unsqueeze(log_action_probs[0][0], 0)))
            
            value, trajs_critic = self.critic_network(in_for_critic, return_dynamics=True, h0=h0_critic)
            #print("\nvalue+trajs", value, trajs_critic)
            
            values = torch.cat((values, value[0][0]))  
            
            h0_actor = trajs  
            h0_critic = trajs_critic

            if info["new_trial"]:
                trial_index = trial_index + 1
                trial_begins.append(time_step+1)
                gt.append(info["gt"])
                errors.append(np.abs(info["gt"]-actions[-2]))
                #h0_actor = torch.zeros(self.hidden_size)
                #h0_critic = torch.zeros(self.hidden_size)

            time_step = time_step + 1

            avarage_error = np.asarray(errors).sum() / n_trs

        observations = np.asarray(observations)
        rewards = np.asarray(rewards)
        actions = np.asarray(actions)
        log_probs = log_probs[1:]
      
        return observations, rewards, actions, log_probs, values, trial_begins, gt, avarage_error
        #      array, array, array, tensor(t_steps, actions), tensor(t_steps), list, list, list
   
   
    
    def learning(self, n_trs, lr_a=1e-4, lr_c=1e-4): 
        
        # TODO
        #if clip_gradient is not None:
        #        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient) 
        
        begin = time.time()
        
        optimizer_actor = torch.optim.Adam(self.actor_network.parameters(), lr=lr_a)
        optimizer_critic = torch.optim.Adam(self.critic_network.parameters(), lr=lr_c)

        #with torch.no_grad():        
        
        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        
        observations, rewards, actions, log_probs, values, trial_begins, gt, avarage_error = self.experience(n_trs)

        cum_rho = np.zeros(0)
        tau_r = np.inf  # Song et al. set this value to 10s only for reaction time tasks
        trial_total_reward = []
        
        for i in range(n_trs):
            
            start = int(trial_begins[i])
            stop = int(trial_begins[i+1])
            
            trial_rewards = rewards[start:stop]
            trial_cumulative_rewards = []
            
            for j in range(len(trial_rewards)):
                
                disc_rew = [r*np.exp(-(i_r)/tau_r) for i_r, r in enumerate(trial_rewards[j+1:])]
                trial_cumulative_rewards.append(np.sum(disc_rew))
            
            trial_cumulative_rewards = np.array(trial_cumulative_rewards)
            trial_total_reward.append(trial_cumulative_rewards[0])
            cum_rho = np.concatenate((cum_rho, trial_cumulative_rewards))
                    
        cum_rho = torch.Tensor(cum_rho)
        trial_total_reward = np.asarray(trial_total_reward)
        
        loss = self.loss(log_probs, actions, cum_rho - values, n_trs)
        loss.backward(retain_graph=True)
        optimizer_actor.step()
        
        loss_mse = self.loss_mse(values, cum_rho)
        loss_mse.backward()
        optimizer_critic.step()
        
        TIME = time.time()-begin
        
        #obs = observations.T
        #length = 0
        #for i in range(n_trs):
#
        #    plt.figure(figsize=(25,5))
#
        #    start = int(trial_begins[i])
        #    stop = int(trial_begins[i+1])
#
        #    plt.plot(obs[0][start:stop], "-o", label="input_fix", color="gray")
        #    plt.plot(obs[1][start:stop], "-o", label="input_right", color="black")
        #    plt.plot(obs[2][start:stop], "-o", label="input_left", color="dimgray")
        #    plt.plot(actions[start:stop], "-o", label="actions", color="red", alpha=0.5)
        #    plt.plot(rewards[start:stop], "-o", label="rewards", color="blue", alpha=0.5)
        #    plt.plot(stop-length-2, gt[i], "*", markersize=25, color="orange")
        #    length += len(obs[0][start:stop])
#
        #    plt.xlabel("t", size=20)
        #    plt.xticks(size=20)
        #    plt.yticks(size=20)
        #    plt.title("trial %i" %(i+1), size=20)
        #    plt.legend(fontsize=15, loc="upper left");
        
        return loss, loss_mse, TIME, trial_total_reward, avarage_error
    
    
    
    def training(self, n_trs, iterations, lr_a=1e-3, lr_c=1e-3):
   
        begin = time.time()
        
        average_time = 0
        actor_rewards = []
        critic_losses = []
        errors = []
    
        for i in range(iterations):
            print(i)
            loss, loss_mse, TIME, trial_total_rewards, avarage_error = self.learning(n_trs, lr_a, lr_c)
            average_time = average_time + TIME
            actor_rewards.append(trial_total_rewards.sum()/n_trs)
            critic_losses.append(loss_mse.detach().numpy())
            errors.append(avarage_error)
            
        print("It took %.2f s for %i iterations\n" %(time.time()-begin, iterations))
        print("It took %.2f s on average for each %i-trails iteration\n" %(average_time/iterations, n_trs))
        
        actor_rewards = np.asarray(actor_rewards)
        critic_losses = np.asarray(critic_losses)
        
        return actor_rewards, critic_losses, errors