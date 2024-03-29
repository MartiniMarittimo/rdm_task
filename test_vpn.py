import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import torch
import copy
import random

import FullRankRNN as rnn
import Reinforce as rln


reinforce = rln.REINFORCE(seed=0)

iterations = 3000
n_trs = 15
lr_a = 1e-4
lr_c = 1e-4

reinforce.training(n_trs=n_trs, epochs=iterations, lr_a=lr_a, lr_c=lr_c, cuda=False,
                   train_actor=True, train_critic= True)
