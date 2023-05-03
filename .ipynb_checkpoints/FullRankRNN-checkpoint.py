import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor
import random
import time



def loss_mse(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    
    # Compute loss for each trial & timestep (average accross output neurons)
    loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    #print("loss_tensor: ", loss_tensor.shape)

    # Compute loss for each trial (average across timesteps)
    # and account also for different number of masked values per trial
    loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    #print("loss_by_trial: ", loss_by_trial.shape)
    
    #Compute the final loss (average across trials)
    loss = loss_by_trial.mean()
    #print("loss: ", loss.shape)
    
    return loss


"""
def train(net, input, n_epochs, lr=1e-2, batch_size=32, plot_learning_curve=False, plot_gradient=False,
          mask_gradients=False, clip_gradient=None, keep_best=False, cuda=False, resample=False, lambda1=None, lambda2=None,
          save_loss=False):
    Train a network
    :param net: nn.Module
    :param input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :param resample: for SupportLowRankRNNs, set True
    :return: nothing
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    ##### CUDA management #####
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    net.to(device=device)
    input = input.to(device=device)
    target = target.to(device=device)
    mask = mask.to(device=device)
    ######
    
    num_samples = input.shape[0] # NUMBER of trials, NUMERO totale di campionamenti i (samples) nel dataset D
    all_losses = []
    mean_losses = []
    if plot_gradient:
        gradient_norms = []
        
    with torch.no_grad():
        output = net(input)  # forward (in this first case using to the whole input dataset)
        initial_loss = loss_mse(output, target, mask)
        #tensor.item() returns the value of this tensor as a standard Python number; it only works for tensors with one element
        mean_losses.append(initial_loss.item()) 
        print("Initial loss: %.3f." % (initial_loss.item()))
        begin = time.time()
        if keep_best:
            best = net.clone()
            best_loss = initial_loss.item()

    for epoch in range(n_epochs):
        losses = []
        
        for mb in range(num_samples // batch_size): # mb numera i mini-batch --> in un'epoca ciclo su tutti i mini-batch
            optimizer.zero_grad()
            random_batch_idx = random.sample(range(num_samples), batch_size)
            batch = input[random_batch_idx]
            if lambda1 is not None:
                output = net(batch) # forward 
                loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx]) + \
                lambda1*torch.sum(torch.abs(net.wrec))/net.wrec.shape[0]**2
            elif lambda2 is not None:
                output, h = net.forward(batch,return_dynamics = True) # forward
                r = net.non_linearity(h)
                mean_abs_rate = torch.sum(r)/r.shape[0]/r.shape[1]/r.shape[2]
                loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx]) + \
                lambda2*mean_abs_rate
            else:
                output = net(batch) # forward
                loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx]) 
            
            losses.append(loss.item())
            all_losses.append(loss.item())
            
            # qui sto calcolando i gradienti quindi dL/dw_previous, con L funzione di costo: 
            # ho un aggiornamento per ogni mini-batch, per ogni epoca
            loss.backward() 
            
            if clip_gradient is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
            
            if plot_gradient:
                tot = 0
                for param in [p for p in net.parameters() if p.requires_grad]:
                    tot += (param.grad ** 2).sum()
                gradient_norms.append(sqrt(tot))
            
            # qui invece sto facendo l'update dei parametri, con i gradienti calcolati precedentemente:
            # ho un aggiornamento per ogni mini-batch, per ogni epoca
            optimizer.step()
            
            # Two important lines to prevent memory leaks
            loss.detach_()
            output.detach_()
            if resample:
                net.resample_basis()
        
        if keep_best and np.mean(losses) < best_loss:
            best = net.clone()
            best_loss = np.mean(losses)
            #print("epoch %d:  loss=%.3f  (took %.2f s) *" % (epoch, np.mean(losses), time.time() - begin))
        #else:
            #print("epoch %d:  loss=%.3f  (took %.2f s)" % (epoch, np.mean(losses), time.time() - begin))
        
        if save_loss is not None:
            mean_losses.append(np.mean(losses))

    print("Final loss: %.3f. It took %.2f s for %d epochs.\n" % (np.mean(losses), time.time() - begin, n_epochs))

    if save_loss is not None:
        #torch.save(all_losses,'../models/all_'+save_loss+'.pt')
        torch.save(mean_losses,'../models/'+save_loss+'.pt')

    if plot_learning_curve:
        plt.plot(all_losses)
        plt.title("Learning curve")
        plt.show()

    if plot_gradient:
        plt.plot(gradient_norms)
        plt.title("Gradient norm")
        plt.show()

    if keep_best:
        net.load_state_dict(best.state_dict())
"""

        
class FullRankRNN(nn.Module): # FullRankRNN is a child class, nn.Module is the parent class

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha=0.2, rho=1, train_wi=False, train_wrec=True,
                 train_wo=False, train_h0=False, wi_init=None, wrec_init=None, wo_init=None, si_init=None, so_init=None):
        """
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float
        :param rho: float, std of gaussian distribution for initialization
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param wrec_init: torch tensor of shape (hidden_size, hidden_size)
        :param si_init: input scaling, torch tensor of shape (input_dim)
        :param so_init: output scaling, torch tensor of shape (output_dim)
        """
        
        super(FullRankRNN, self).__init__()  #???
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rho = rho
        self.train_wi = train_wi #boolean (False)
        self.train_wrec = train_wrec #boolean (True)
        self.train_wo = train_wo #boolean (False)
        self.train_h0 = train_h0 #boolean (False)
        self.non_linearity = torch.nn.ReLU()

        # torch.nn.Parameter(tensor) is a kind of Tensor that is to be considered a module parameter.
        # Parameters are Tensor subclasses, that have a very special property when used with Modules: 
        # when they’re assigned as Module attributes, they are automatically added to the list of its parameters, 
        # and will appear e.g. in parameters() iterator. Assigning a Tensor doesn’t have such effect. 
        # Parameter has a method called requires_grad which is a boolean value and assesses whether the parameter requires 
        # gradient or not. It is optional and its default value is True.

        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size)) #matrice 2D
        self.si = nn.Parameter(torch.Tensor(input_size)) #vettore 1D
        if train_wi:
            self.si.requires_grad = False 
        if not train_wi:
            self.wi.requires_grad = False
            
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wrec:
            self.wrec.requires_grad = False
            
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.so = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.so.requires_grad = False
        if not train_wo:
            self.wo.requires_grad = False
            
        self.h0 = nn.Parameter(torch.Tensor(hidden_size)) 
        if not train_h0:
            self.h0.requires_grad = False

        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(wi_init)
            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)
            if wrec_init is None:
                self.wrec.normal_(std = rho/sqrt(hidden_size))
            else:
                self.wrec.copy_(wrec_init)
            if wo_init is None:
                self.wo.normal_(std = 5/hidden_size)
            else:
                self.wo.copy_(wo_init)
            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)
            self.h0.zero_() #fills self tensor with zeros
        self.wi_full, self.wo_full = [None] * 2
        self.define_proxy_parameters()
    
        
    def define_proxy_parameters(self):
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so
        

    def forward(self, input, return_dynamics=False):       
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        IMPORTANT --> the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :return: if return_dynamics=False, output tensor of shape(batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, output tensor & trajectories tensor of shape(batch_size, #timesteps, #hidden_units)
        """
        
        #batch_size = input.shape[0]
        #seq_len = input.shape[1]
        h = self.h0 
        r = self.non_linearity(h)
        self.define_proxy_parameters()
        input = torch.Tensor(input)
        
        #inizializzazione del rumore interno alla rete, gaussiana N(0,1)
        noise = torch.randn(self.hidden_size, device=self.wrec.device) 
        output = torch.zeros(self.output_size, device=self.wrec.device)
        
        if return_dynamics:
            trajectories = torch.zeros(self.hidden_size, device=self.wrec.device)

        # forward loop: it's a way to integrate in time
        h = h + self.alpha * (- h + input.matmul(self.wi_full) + r.matmul(self.wrec.t())) + self.noise_std * noise 
        r = self.non_linearity(h)
        output = r.matmul(self.wo_full)
        output = output.detach().numpy()
        
        #SOFTMAX
        output = np.exp(output)
        denom = output.sum()
        for i in range(len(output)):
            num = output[i]
            output[i] = num / denom
        
        # TEST
        output = np.array([0.8,0.1,0.1])
        
        if return_dynamics:
            trajectories = r
            return output, trajectories

        else:
            return output
        

    #def clone(self):
    #    new_net = FullRankRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
    #                          self.rho, self.train_wi, self.train_wrec, self.train_wo, self.train_h0,
    #                          self.wi, self.wrec, self.wo, self.si, self.so)
    #    return new_net


