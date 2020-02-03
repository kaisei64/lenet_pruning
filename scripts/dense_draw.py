import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from channel_mask_generator import ChannelMaskGenerator
from dense_mask_generator import DenseMaskGenerator
from dataset import *
from pfgacnn import PfgaCnn
from cnn_evaluateprune import CnnEvaluatePrune
from result_save_visualization import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from draw_architecture import mydraw

original_net = parameter_use(f'./result/pkl1/dense_prune_90per.pkl')

original_dense_list = [module for module in original_net.modules() if isinstance(module, nn.Linear)]

mydraw([torch.t(original_dense_list[0].weight).cpu().detach().numpy(), torch.t(original_dense_list[1].weight).cpu().detach().numpy(),
        torch.t(original_dense_list[2].weight).cpu().detach().numpy()])
