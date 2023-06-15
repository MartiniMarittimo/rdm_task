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
                 deltaT=20., noise_std=0, alpha=0.2, name_load_actor=None, name_load_critic=None,
                 train_wi_a=True, train_wrec_a=True, train_wo_a=True,
                 train_wi_c=True, train_wrec_c=True, train_wo_c=True):        
        
        self.actor_network = rnn.FullRankRNN(input_size, hidden_size, output_size,
                                             noise_std=noise_std, alpha=alpha, rho=0.8,
                                             train_wi=train_wi_a, train_wo=train_wo_a, train_wrec=train_wrec_a)
        
        if name_load_actor is not None:
            self.actor_network.load_state_dict(torch.load(name_load_actor, map_location='cpu'))
        
        self.actor_network.actor_critic(actor=True)
        
        self.critic_network = rnn.FullRankRNN(input_size=129, hidden_size=128, output_size=1,
                                              noise_std=noise_std, alpha=alpha, rho=0.8,
                                              train_wi=train_wi_c, train_wo=train_wo_c, train_wrec=train_wrec_c)
        if name_load_critic is not None:
            self.critic_network.load_state_dict(torch.load(name_load_critic, map_location='cpu'))

        self.critic_network.actor_critic(actor=False)
        
        self.task = rdm.RandomDotMotion(dt=deltaT)
        
        self.hidden_size = hidden_size
        
        self.coh_info = {"n0":0, "r0":0, "pos0": 0, "neg0": 0, "n6":0, "r6":0, "pos6": 0, "neg6": 0,\
                         "n12":0, "r12":0, "pos12": 0, "neg12": 0, "n25":0, "r25":0, "pos25": 0, "neg25": 0,\
                         "n51":0, "r51":0, "pos51": 0, "neg51": 0}
        self.trial = 0
        self.iteration = 0
        
# ===============================================================================================================
        
    def loss(self, log_probs, actions, cum_rho, values, n_trs):
        
        new_mask = torch.zeros(log_probs.size())
        
        for i in range(len(actions)):
            action = actions[i]
            new_mask[i][action] = 1
            
        #assert torch.all(torch.eq(new_mask, full_mask))
        
        loss = (new_mask * log_probs)
        loss = loss.sum(dim=-1)
        loss = loss * (cum_rho - values)        
        loss = loss.sum(dim=-1) / (-n_trs)
        
        return loss
    
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

    def experience(self, n_trs, coh_info=None):
        
        begin = time.time()

        observations = []
        rewards = []
        
        log_probs = torch.unsqueeze(torch.zeros(3), 0)
        actions = []
        
        inputs = []
        values = torch.zeros(0) 
        
        gt = []
        coh = []
        errors = []
        if coh_info is None:
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
            
            ob, rwd, done, info = self.task.step(action=action)
            observations.append(ob)
            rewards.append(rwd)
            
            ob = torch.Tensor(np.array([ob]))
            ob = torch.unsqueeze(ob, 0) # tensor of size (1,1,3)
                
            action_probs, trajs = self.actor_network(ob, return_dynamics=True, h0=h0_actor) 
            
            p = action_probs[0][0].clone().detach().numpy()
            action = np.random.choice(np.arange(len(p)), p=p) # 0, 1, 2: fix, right, left
            actions.append(action)
            action_t = torch.Tensor([action])
            relu_trajs = self.actor_network.non_linearity(trajs[0][0])
            in_for_critic = torch.unsqueeze(torch.unsqueeze(torch.cat((action_t, relu_trajs.detach())),0),0)
#            in_for_critic = torch.unsqueeze(ob[:, :, 0],0)
#            inputs.append(in_for_critic.item())
            log_action_probs = torch.log(action_probs)
            log_probs = torch.cat((log_probs, torch.unsqueeze(log_action_probs[0][0], 0)))
            
            value, trajs_critic = self.critic_network(in_for_critic, return_dynamics=True, h0=h0_critic)
            values = torch.cat((values, value[0][0]))  
            
            h0_actor = trajs  
            h0_critic = trajs_critic

            if info["new_trial"]:
                self.trial += 1
                trial_index = trial_index + 1
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
   
# =============================================================================================================== 
         
    def update_actor(self, optimizer_actor, lr_a, loss):#log_probs, actions, cum_rho, values, n_trs):
        
        optimizer_actor.zero_grad()
        loss.backward()#retain_graph=True)
        optimizer_actor.step()

# =============================================================================================================== 

    def update_critic(self, optimizer_critic, lr_c, loss_mse):#cum_rho, values, trial_begins, n_trs):
    
        optimizer_critic.zero_grad()
        loss_mse.backward()
        optimizer_critic.step()

# =============================================================================================================== 
       
    def learning_step(self, optimizer_actor, optimizer_critic, epoch, n_trs, coh_info=None, lr_a=1e-4, lr_c=1e-4, plot=False): 
        
        # TODO
        #if clip_gradient is not None:
        #        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient) 
        
        begin = time.time() 

        #with torch.no_grad():        
        
        if coh_info is None:
            coh_info = self.coh_info
        
        observations, rewards, actions, log_probs, values,\
        trial_begins, gt, coh, avarage_error, inputs = self.experience(n_trs, coh_info=coh_info)
        
        wi = self.critic_network.wi.data
        wo = self.critic_network.wo.data

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
        
        rewards = torch.Tensor(np.asarray(rewards))
        
        #QUI NON MOLTO CHIARO DEL PERCHÈ DEBBA FARE DETACH() DAI VALUES... 
        #È VERO CHE LI OTTENGO COME OUTPUT DI UNA RETE NEURALE CON PARAMETRI PHI, 
        #MA È ANCHE VERO CHE ALL'ACTOR_PTIMIZER IO PASSO ESPLICITAMENTE SOLO I PARAMETRI THETA
        #QUANDO LO INIZIALIZZO DENTRO LA TRAINING FUNCTION
        loss = self.loss(log_probs, actions, cum_rho, values.clone().detach().numpy(), n_trs) 
        self.update_actor(optimizer_actor, lr_a, loss)
        loss_mse = self.loss_mse(cum_rho, values, trial_begins, n_trs)
        self.update_critic(optimizer_critic, lr_c, loss_mse)
        
        if epoch < 10 or epoch >= self.iterations-10:
            torch.save(cum_rho, "debug_data/cum_rho"+str(epoch)+".pt")
            torch.save(values, "debug_data/values"+str(epoch)+".pt")
            torch.save(observations, "debug_data/observations"+str(epoch)+".pt")
        
        TIME = time.time()-begin
        
        if plot:
            obs = observations.T
            length = 0
            for i in range(n_trs):

                plt.figure(figsize=(25,5))

                start = int(trial_begins[i])
                stop = int(trial_begins[i+1])

                plt.plot(obs[0][start:stop], "-o", label="input_fix", color="gray")
                plt.plot(obs[1][start:stop], "-o", label="input_right", color="black")
                plt.plot(obs[2][start:stop], "-o", label="input_left", color="dimgray")
                plt.plot(actions[start:stop], "-o", label="actions", color="red", alpha=0.5)
                plt.plot(rewards[start:stop], "-o", label="rewards", color="blue", alpha=0.5)
                plt.plot(cum_rho[start:stop], "-o", label="cum_rho", color="cyan", alpha=0.5)
                plt.plot(values.clone().detach().numpy()[start:stop], "-o", label="values", color="purple", alpha=0.5)
                plt.plot(stop-length-2, gt[i], "*", markersize=25, color="orange")
                plt.axhline(0, color="black")
                length += len(obs[0][start:stop])

                plt.xlabel("t", size=20)
                plt.xticks(size=20)
                plt.yticks(size=20)
                plt.text(15, 0.5, 'coh=%i, %s' %(coh[i], gt[i]), style='italic', fontsize=20,\
                         bbox={'facecolor': 'grey', 'alpha': 0.3, 'pad': 10})
                plt.title("trial %i, epoch %i" %(i+1, epoch), size=20)
                plt.legend(fontsize=15, loc="upper left");
        
        return loss, loss_mse, TIME, trial_total_reward, avarage_error, wi, wo
    
# ===============================================================================================================
    
    def training(self, n_trs, iterations, lr_a=1e-3, lr_c=1e-3):
   
        begin = time.time()
        
        average_time = 0
        actor_rewards = []
        critic_losses = []
        errors = []
        w_i = []
        w_o = []
        plot = False
        coh_info = self.coh_info
        self.iterations = iterations
        epoch = 0
        optimizer_actor = torch.optim.Adam(self.actor_network.parameters(), lr=lr_a)
        optimizer_critic = torch.optim.Adam(self.critic_network.parameters(), lr=lr_c)

        for i in range(iterations):      
            epoch = i
            if i == iterations-1:
                plot = True
            if (i+1)%10 == 0:
                print("iteration", i+1)    
            loss, loss_mse, TIME, trial_total_rewards,\
            avarage_error, wi, wo = self.learning_step(optimizer_actor, optimizer_critic, epoch, n_trs, coh_info, lr_a, lr_c, plot)
            average_time = average_time + TIME
            actor_rewards.append(trial_total_rewards.sum()/n_trs)
            critic_losses.append(loss_mse.detach().numpy())
            errors.append(avarage_error)
            #print(wi, "\n")
            #print(wo, "\n")
            #w_i.append(wi.item())
            #w_o.append(wo.item())
        torch.save(self.actor_network.state_dict(), "models/RL_actor_network.pt".format(self.hidden_size))
        torch.save(self.critic_network.state_dict(), "models/RL_critic_network.pt".format(self.hidden_size))
            
        print("It took %.2f m for %i iterations\n" %((time.time()-begin)/60, iterations))
        print("It took %.2f s on average for each %i-trails iteration\n" %(average_time/iterations, n_trs))
        
        actor_rewards = np.asarray(actor_rewards)
        critic_losses = np.asarray(critic_losses)
        
        return actor_rewards, critic_losses, errors, coh_info, w_i, w_o