import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import seaborn as sns
import pytorch_lightning as pl
from matplotlib.colors import to_rgb
from scipy import signal as ss
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from rqs import unconstrained_rqs,rqs
from GatedResNet import *


''' Building a coupling layer '''

class CouplingLayer(nn.Module):
    def __init__(self,network,mask,in_channels,K=3,B:int=8,shape=28):
        '''
        Coupling Layer inside the normalizing flow
        Inputs:
            network     - Pytroch neural network to be used inside the flow (Something like ResNet or PixelCNN).
            mask        - Binary mask of 0 and 1 which tells us if that entry should be used by the NN or not.
            in_channels - Number of input channels.
        '''
        super().__init__()
        self.network = network
        self.scaling = nn.Parameter(data=torch.zeros(in_channels))
        self.register_buffer('mask', mask)
        self.K = K
        self.B = B
        self.shape=shape

    def forward(self,z,log_det,reverse=False):
        '''
        Inputs:
            z - Latent input ot the flow
            log_det - The current log determinant from the previous flow.
                      The log determinant of this layer will be added to this one.
            reverse - If True we apply the inverse of the layer.
        '''
        z_in   = z * self.mask
        z_out  = z * (- ~self.mask.to(torch.bool).to(torch.int32))

        if not reverse:
            nn_out = self.network(z_in)
            inputs_in = z_in.view(z_in.shape[0],-1)
            inputs_out = z_out.view(z_in.shape[0],-1)
            # in_in, in_out = [784,1]
            W,H,D  = torch.split(nn_out,self.K,dim=-1)
            #W.size = [1,784,3]
            '''
            Vectors theta^W and theta^H are passed through softmax
            and multiplied by 2B.
            '''
            W,H = torch.softmax(W,dim=-1),torch.softmax(H,dim=-1)
            W,H = (2*self.B) * W, (2*self.B) * H

            '''
            D is passed through softplus - interpreted as derivatives
            '''
            D = F.softplus(D)
            z_out,ld = unconstrained_rqs(inputs_out,W,H,D,self.shape,self.B)
            print("Size")
            print(ld.size())
            print(z_out.size())
            print("Size")
            log_det += torch.sum(ld,dim=1)
            out_nr = self.network(z_out)
            W,H,D  = torch.split(out_nr,self.K,dim=-1)
            W,H = torch.softmax(W,dim=-1),torch.softmax(H,dim=-1)
            W,H = (2*self.B) * W, (2*self.B) * H
            D = F.softplus(D)

            z_in, ldz = unconstrained_rqs(inputs_in,W,H,D,self.shape,self.B)
            log_det += torch.sum(ldz,dim=1)
            z = z_in + z_out
            print("Finished if ")
            return z, log_det
        else:
            print("IN THE ELSE")
            nn_out = self.network(z_out)
            W,H,D  = torch.split(nn_out,self.K,dim=-1)
            W,H = torch.softmax(W,dim=-1),torch.softmax(H,dim=-1)
            W,H = (2*self.B) * W, (2*self.B) * H
            D = F.softplus(D)

            inputs_in = z_in.reshape(self.shape,self.shape)
            inputs_out = z_out.reshape(self.shape,self.shape)

            z_in,ld = unconstrained_rqs(inputs_in,W,H,D,self,shape,self.B,inverse=True)

            log_det += torch.sum(ld,dim=1)
            out = self.network(z_in).reshape(self.shape,self.shape,-1)

            W,H,D  = torch.split(out,self.K,dim=-1)
            W,H = torch.softmax(W,dim=-1),torch.softmax(H,dim=-1)
            W,H = (2*self.B) * W, (2*self.B) * H
            D = F.softplus(D)

            z_out, ld = unconstrained_rqs(inputs_out,W,H,D,self.shape,self.B,inverse=True)
            log_det += torch.sum(ld,dim=1)

            z = z_in + z_out
            print("passed else")
            return z, log_det

def checkerBoardMask(h,w,inverse=False):
    x,y   = torch.arange(w,dtype=torch.int32),torch.arange(h,dtype=torch.int32)
    xx,yy = torch.meshgrid(x,y)

    mask = torch.fmod(xx+yy,2).to(torch.float32)

    if inverse:
        mask = 1 - mask
    return mask


'''
dim1=10
dim0=40
batch_size=150
x = torch.rand(batch_size,1,dim0,dim1)
K = 3
ld = torch.rand(dim0)
cl = CouplingLayer(network=GatedResNet(1,16,K*3-1,batch_size=batch_size),
              mask=checkerBoardMask(w=dim0,h=dim1,inverse=False),
              in_channels=1,shape=dim0)

cl(x,ld)

t = torch.rand((1,1,10,10))
mask = checkerBoardMask(10,10)
print(mask)
t = t * mask
print(t)
mask2 =checkerBoardMask(10,10,True)
print(mask2)
'''
