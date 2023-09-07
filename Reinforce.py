import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import copy
import time

import FullRankRNN as rnn
import RDMtask as rdm



class REINFORCE:
    
    def __init__(self, deltaT=20., noise_std=0, alpha=0.2, 
                 name_load_actor=None, name_load_critic=None, seed=None,
                 train_wi_a=True, train_wrec_a=True, train_wo_a=True,
                 train_wi_c=True, train_wrec_c=True, train_wo_c=True):        

        self.cuda = False
        self.device = torch.device('cpu')

        if seed is not None:
            torch.manual_seed(seed)
        self.actor_network = rnn.FullRankRNN(input_size=3, hidden_size=128, output_size=3,
                                             noise_std=noise_std, alpha=alpha, rho=0.8,
                                             train_wi=train_wi_a, train_wo=train_wo_a, train_wrec=train_wrec_a)
        
        if name_load_actor is not None:
            self.actor_network.load_state_dict(torch.load(name_load_actor, map_location=self.device))
        
        self.actor_network.actor_critic(actor=True)
        
        if seed is not None:
            torch.manual_seed(seed)
        self.critic_network = rnn.FullRankRNN(input_size=131, hidden_size=128, output_size=1,
                                              noise_std=noise_std, alpha=alpha, rho=0.8,
                                              train_wi=train_wi_c, train_wo=train_wo_c, train_wrec=train_wrec_c)
        if name_load_critic is not None:
            self.critic_network.load_state_dict(torch.load(name_load_critic, map_location=self.device))

        self.critic_network.actor_critic(actor=False)
        
        self.task = rdm.RandomDotMotion(dt=deltaT)
        
        self.hidden_size = 128
        
        self.coh_info = {"n0":0, "r0":0, "pos0": 0, "neg0": 0, "n6":0, "r6":0, "pos6": 0, "neg6": 0,\
                         "n12":0, "r12":0, "pos12": 0, "neg12": 0, "n25":0, "r25":0, "pos25": 0, "neg25": 0,\
                         "n51":0, "r51":0, "pos51": 0, "neg51": 0}
        self.trial = 0
        self.epochs = 0
        self.actions_t = torch.zeros(3, device=self.device)
        self.actions_tt = []

        
# ===============================================================================================================
        
    def obj_function(self, log_action_probs, actions, cum_rho, values, n_trs):
        
        new_mask = torch.zeros(log_action_probs.size(), device=self.device)
        
        for i in range(len(actions)):
            action = actions[i]
            new_mask[i][action] = 1
            
        #assert torch.all(torch.eq(new_mask, full_mask))
        
        obj = (new_mask * log_action_probs)
        obj = obj.sum(dim=-1)
        obj = obj * (cum_rho - values)        
        obj = obj.sum(dim=-1) / (-n_trs)
        
        return obj
    
# =============================================================================================================== 
    
    def loss_mse(self, output, target, trial_begins, n_trs):
        
        loss = 0
        
        for i in range(n_trs):
            
            start = int(trial_begins[i])
            stop = int(trial_begins[i+1])
            T = stop - start
            
            trial_output = output[start:stop]
            trial_target = target[start:stop]
            
            L = (trial_output - trial_target).pow(2).sum(dim=-1) / T
            loss = loss + L
            
        loss = loss / n_trs

        return loss

# =============================================================================================================== 
    
    def experience(self, n_trs, training=False):
        
        device = self.device
        
        observations = []
        rewards = []
        
        log_action_probs = torch.unsqueeze(torch.zeros(3, device=device), 0)
        actions = []
        
        values = torch.zeros(0, device=device)
        
        gt = []
        coh = []
        avarage_error = 0
        
        trial_index = 0
        time_step = -1
        trial_begins = [0]

        self.task.reset()
        action = 0
        
        h0_actor = torch.zeros(self.hidden_size, device=device)
        h0_critic = torch.zeros(self.hidden_size, device=device)

        while trial_index < n_trs: #ciclo su tutti i time-step in fila di tutti gli n_trs trials
            
            time_step += 1
            
            ob, rwd, done, info = self.task.step(action=action)
            observations.append(ob)
            rewards.append(rwd)
            ob = torch.tensor(np.array([ob]), device=device)
            ob = torch.unsqueeze(ob, 0) # tensor of size (1,1,3)
            #ob = ob.to(device=device)
            
            action_probs, trajs = self.actor_network(ob, return_dynamics=True, h0=h0_actor) 
            #action_probs = action_probs.to(device=torch.device('cpu'))
            log_probs = torch.log(action_probs)
            log_action_probs = torch.cat((log_action_probs, torch.unsqueeze(log_probs[0][0], 0)))
            
            p = action_probs[0][0].clone().detach().to(device=torch.device('cpu')).numpy()
            action = np.random.choice(np.arange(len(p)), p=p) # 0, 1, 2: fix, right, left
            actions.append(action) 
            
            if action == 0:
                self.actions_t[0] = 1
            elif action == 1:
                self.actions_t[1] = 1
            elif action == 2:
                self.actions_t[2] = 1
            self.actions_tt.append(self.actions_t.clone())
            action_t = torch.tensor([action],device=device)
            relu_trajs = self.actor_network.non_linearity(trajs[0][0])
            in_for_critic = torch.unsqueeze(torch.unsqueeze(torch.cat((self.actions_t, relu_trajs.detach())),0),0)
            self.actions_t.zero_()
#            in_for_critic = torch.unsqueeze(torch.unsqueeze(relu_trajs.detach(),0),0)
            
            value, trajs_critic = self.critic_network(in_for_critic, return_dynamics=True, h0=h0_critic)
            values = torch.cat((values, value[0][0]))  
            
            h0_actor = trajs  
            h0_critic = trajs_critic

            if info["new_trial"]:
                
                trial_index += 1
                if (trial_index)%100 == 0:
                    print("iteration", trial_index)
                if training:
                    self.trial += 1
                    
                trial_begins.append(time_step+1)
                
                gt.append(info["gt"])
                coh.append(info["coh"])
                if info["performance"] == 0:
                    avarage_error += 1

                if info["coh"] == 0:
                    self.coh_info["n0"] += 1
                    self.coh_info["r0"] += rwd
                    if info["gt"] == 1:
                        self.coh_info["pos0"] = self.trial
                    else:
                        self.coh_info["neg0"] = self.trial
                elif info["coh"] == 6.4:
                    self.coh_info["n6"] += 1
                    self.coh_info["r6"] += rwd
                    if info["gt"] == 1:
                        self.coh_info["pos6"] = self.trial
                    else:
                        self.coh_info["neg6"] = self.trial
                elif info["coh"] == 12.8:
                    self.coh_info["n12"] += 1
                    self.coh_info["r12"] += rwd
                    if info["gt"] == 1:
                        self.coh_info["pos12"] = self.trial
                    else:
                        self.coh_info["neg12"] = self.trial
                elif info["coh"] == 25.6:
                    self.coh_info["n25"] += 1
                    self.coh_info["r25"] += rwd
                    if info["gt"] == 1:
                        self.coh_info["pos25"] = self.trial
                    else:
                        self.coh_info["neg25"] = self.trial
                elif info["coh"] == 51.2:
                    self.coh_info["n51"] += 1
                    self.coh_info["r51"] += rwd
                    if info["gt"] == 1:
                        self.coh_info["pos51"] = self.trial
                    else:
                        self.coh_info["neg51"] = self.trial
                        
                h0_actor.fill_(0)
                h0_critic.fill_(0)
                
        avarage_error /= n_trs

        log_action_probs = log_action_probs[1:]

      
        return observations, rewards, actions, log_action_probs, values, trial_begins, gt, coh, avarage_error
        #      list, list, list, tensor(t_steps, actions), tensor(t_steps), list, list, list, list
         
# =============================================================================================================== 
         
    def update_actor(self, optimizer_actor, obj_function):
        
        obj_function.backward()#retain_graph=True)
        optimizer_actor.step()
        obj_function.detach_()

# =============================================================================================================== 

    def update_critic(self, optimizer_critic, loss_mse):
    
        loss_mse.backward()
        optimizer_critic.step()
        loss_mse.detach_()

# =============================================================================================================== 
       
    def learning_step(self, optimizer_actor, optimizer_critic, epoch, n_trs, train_actor, train_critic): 
        
        begin = time.time() 
        
        device = self.device
        
        # TODO
        #if clip_gradient is not None:
        #        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient) 
        
        #with torch.no_grad():        
        
        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        
        observations, rewards, actions, log_action_probs, values,\
        trial_begins, gt, coh, avarage_error = self.experience(n_trs, training=True)

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
                    
        cum_rho = torch.tensor(cum_rho, device=device)
        trial_total_reward = np.asarray(trial_total_reward)
        
        rewards = torch.Tensor(np.asarray(rewards))
        
        """ QUI NON MOLTO CHIARO DEL PERCHÈ DEBBA FARE DETACH() DAI VALUES... 
            È VERO CHE LI OTTENGO COME OUTPUT DI UNA RETE NEURALE CON PARAMETRI PHI, 
            MA È ANCHE VERO CHE ALL'ACTOR_OPTIMIZER, QUANDO LO INIZIALIZZO DENTRO LA TRAINING FUNCTION,
            GLI PASSO ESPLICITAMENTE SOLO I PARAMETRI THETA 
        """
        actions = np.asarray(actions)
        detached_values = values.clone().detach()#.numpy()
        
        obj_function = self.obj_function(log_action_probs, actions, cum_rho, detached_values, n_trs) 
        if train_actor:
            self.update_actor(optimizer_actor, obj_function)
        
        loss_mse = self.loss_mse(cum_rho, values, trial_begins, n_trs)
        if train_critic:
            self.update_critic(optimizer_critic, loss_mse)
        
        #log_action_probs.detach_() ...?
        #values.detach_() ...?          
        
        return obj_function, loss_mse, trial_total_reward, avarage_error
    
# ===============================================================================================================
    
    def training(self, n_trs, epochs, lr_a=1e-4, lr_c=1e-4, cuda=False, train_actor=True, train_critic=True):
   
        begin = time.time()
    
        self.cuda = cuda
        
        if self.cuda:
            if not torch.cuda.is_available():
                print("Warning: CUDA not available on this machine, switching to CPU")
                self.device = torch.device('cpu')
            else:
                print("Okay with CUDA")
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        device = self.device
        self.actor_network.to(device=device)
        self.critic_network.to(device=device)
        
        optimizer_actor = torch.optim.Adam(self.actor_network.parameters(), lr=lr_a)
        optimizer_critic = torch.optim.Adam(self.critic_network.parameters(), lr=lr_c)
        
        actor_rewards = []
        actor_errors = []
        critic_losses = []
        
        s_o = []
        s_o_critic = []
        
        self.epochs = epochs
        
        copied_actor = copy.deepcopy(self.actor_network.state_dict())
        copied_critic = copy.deepcopy(self.critic_network.state_dict())
        torch.save(copied_actor, "models/RL_actor_network_bef.pt".format(self.hidden_size))
        torch.save(copied_critic, "models/RL_critic_network_bef.pt".format(self.hidden_size))

        for epoch in range(epochs):          
           
            obj_function, loss_mse, trial_total_rewards, avarage_error = self.learning_step(optimizer_actor,
                                                                                            optimizer_critic,
                                                                                            epoch, n_trs,
                                                                                            train_actor,
                                                                                            train_critic)
            actor_rewards.append(trial_total_rewards.sum()/n_trs)
            actor_errors.append(avarage_error)
            critic_losses.append(loss_mse.detach())#.numpy())
        
            s_o.append(self.actor_network.so.data.clone().detach())#.numpy())
            s_o_critic.append(self.critic_network.so.data.clone().detach())#.numpy())
            
            if (epoch+1)%500 == 0 or epoch < 5:
                print("iteration", epoch+1, "- %.2f s so far" %((time.time()-begin)))

        copied_actor2 = copy.deepcopy(self.actor_network.state_dict())
        copied_critic2 = copy.deepcopy(self.critic_network.state_dict())
        torch.save(copied_actor2, "models/RL_actor_network.pt".format(self.hidden_size))
        torch.save(copied_critic2, "models/RL_critic_network.pt".format(self.hidden_size))
        
        torch.save(actor_rewards, 'models/actor_rewards.pt')
        torch.save(actor_errors, 'models/actor_errors.pt')
        torch.save(critic_losses, 'models/critic_loss.pt')
        
        torch.save(s_o, 'models/s_o.pt')
        torch.save(s_o_critic, 'models/s_o_critic.pt')
            
        print("\nDEVICE: " + str(device) + ". It took %.2f m for %i epochs. %i trials per epoch." %((time.time()-begin)/60, epochs, n_trs))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# ===============================================================================================================

    def experience_withCUDA(self, n_trs):
        
        begin = time.time()
        
        if self.cuda:
            if not torch.cuda.is_available():
                device = torch.device('cpu')
            else:
                device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        observations = []
        rewards = []
        
        log_probs = torch.unsqueeze(torch.zeros(3), 0)
        actions = []
        
        inputs = []
        values = torch.zeros(0)
        #values = values.to(device=device) 
        
        gt = []
        coh = []
        errors = []
        coh_info = self.coh_info
        
        n_trs = n_trs
        trial_index = 0
        time_step = 0
        trial_begins = [time_step]

        self.task.reset()
        action = 0
        
        h0_actor = torch.zeros(self.hidden_size)
        h0_critic = torch.zeros(self.hidden_size)

        while trial_index < n_trs:    
            
            #if (trial_index+1)%200 == 0:
            #    print("iteration", trial_index+1)
            
            ob, rwd, done, info = self.task.step(action=action)
            observations.append(ob)
            rewards.append(rwd)
            
            ob = torch.Tensor(np.array([ob]))
            ob = torch.unsqueeze(ob, 0) # tensor of size (1,1,3)
            #ob = ob.to(device=device)
            #h0_actor = h0_actor.to(device=device)
            #h0_critic = h0_critic.to(device=device)

            action_probs, trajs = self.actor_network(ob, return_dynamics=True, h0=h0_actor) 
            #action_probs = action_probs.to(device='cpu')
            p = action_probs[0][0].clone().detach().numpy()
            action = np.random.choice(np.arange(len(p)), p=p) # 0, 1, 2: fix, right, left
            actions.append(action)
            action_t = torch.Tensor([action])
            relu_trajs = self.actor_network.non_linearity(trajs[0][0])
            in_for_critic = torch.unsqueeze(torch.unsqueeze(relu_trajs.detach(),0),0)
#            in_for_critic = torch.unsqueeze(torch.unsqueeze(torch.cat((action_t, relu_trajs.detach())),0),0)
#            in_for_critic = torch.unsqueeze(ob[:, :, 0],0)
#            inputs.append(in_for_critic.item())
            log_action_probs = torch.log(action_probs)
            log_probs = torch.cat((log_probs, torch.unsqueeze(log_action_probs[0][0], 0)))
            
            value, trajs_critic = self.critic_network(in_for_critic, return_dynamics=True, h0=h0_critic)
            #value = value.to(device='cpu')
            values = torch.cat((values, value[0][0]))  
            
            h0_actor = trajs  
            h0_critic = trajs_critic

            if info["new_trial"]:
                self.trial += 1
                trial_index = trial_index + 1
                if (trial_index+1)%100 == 0:
                    print("iteration", trial_index+1)
                trial_begins.append(time_step+1)
                gt.append(info["gt"])
                coh.append(info["coh"])
                if info["coh"] == 0:
                    coh_info["n0"] += 1
                    coh_info["r0"] += rwd
                    if info["gt"] == 1:
                        coh_info["pos0"] = self.trial
                    else:
                        coh_info["neg0"] = self.trial
                elif info["coh"] == 6.4:
                    coh_info["n6"] += 1
                    coh_info["r6"] += rwd
                    if info["gt"] == 1:
                        coh_info["pos6"] = self.trial
                    else:
                        coh_info["neg6"] = self.trial
                elif info["coh"] == 12.8:
                    coh_info["n12"] += 1
                    coh_info["r12"] += rwd
                    if info["gt"] == 1:
                        coh_info["pos12"] = self.trial
                    else:
                        coh_info["neg12"] = self.trial
                elif info["coh"] == 25.6:
                    coh_info["n25"] += 1
                    coh_info["r25"] += rwd
                    if info["gt"] == 1:
                        coh_info["pos25"] = self.trial
                    else:
                        coh_info["neg25"] = self.trial
                elif info["coh"] == 51.2:
                    coh_info["n51"] += 1
                    coh_info["r51"] += rwd
                    if info["gt"] == 1:
                        coh_info["pos51"] = self.trial
                    else:
                        coh_info["neg51"] = self.trial
                errors.append(np.abs(info["gt"]-actions[-2]))
                h0_actor = torch.zeros(self.hidden_size)
                h0_critic = torch.zeros(self.hidden_size)

            time_step = time_step + 1

            avarage_error = np.asarray(errors).sum() / n_trs

        observations = np.asarray(observations)
        actions = np.asarray(actions)
        log_probs = log_probs[1:]

      
        return observations, rewards, actions, log_probs, values, trial_begins, gt, coh, avarage_error, inputs
        #      array, array, array, tensor(t_steps, actions), tensor(t_steps), list, list, list, list
   
        
        
