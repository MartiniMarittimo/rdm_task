import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor
import random
import time


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
        self.actor = False

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
    
    
    
    def actor_critic(self, actor=False):
        
        if actor:
            self.actor = True
        
        
        
    def forward(self, input, h0, return_dynamics=False, actor=False):       
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        IMPORTANT --> the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :return: if return_dynamics=False, output tensor of shape(batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, output tensor & trajectories tensor of shape(batch_size, #timesteps, #hidden_units)
        """
        
        h = h0 
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
        
        if self.actor and return_dynamics:            
            #SOFTMAX
            #print(output)
            #print(output[0][0])
            soft_output = output.clone()
            for i in range(len(output[0][0])):
                #print(output[0][0]-output[0][0][i])
                exp_output = torch.exp(output[0][0]-output[0][0][i])
                denom = exp_output.sum()
                soft_output[0][0][i] = 1 / denom
            
            trajectories = h
            return soft_output, trajectories, output, exp_output, denom

        
        # TEST
        #output = torch.Tensor([0.8,0.1,0.1])
        
        if return_dynamics:
            trajectories = h
            return output, trajectories

        else:
            return output