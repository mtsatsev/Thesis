import typing
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.optim as optim

class LayerNorm(nn.Module):
    '''
    Applies normalization accross all input channels.
    '''
    def __init__(self,in_channels):
        '''
        in_channels: Number of input channels.
        '''
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)

    def forward(self,x):
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = x.permute(0,3,1,2)


class MaskConv(nn.Conv2d):
    def __init__(self,mask_type:bool):
        super().__init__()
        self.register_buffer('mask',torch.zeros_like(self.weights))
        self.create_mask()

    def create_mask(self,mask_type:bool):
        k = self.kernel_size[0]
        self.mask[:,:,:k//2] = 1
        self.mask[:,:,k//2,:k//2] = 1
        if mask_type:
            self.mask[:,:,k//2,k//2] = 1

    def forward(self,x):
        batch = x.shape[0]
        y = F.conv2d(x,self.weights*self.mask, self.bias, self.stride, self.padding,self.dilation,self.groups)
        return y

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.ModuleList([
            nn.ReLU(),
            MaskConv(True,in_channels,in_channels//2,1),
            nn.ReLU(),
            MaskConv(True,in_channels//2,in_channels//2,7,padding=3),
            nn.ReLU(),
            MaskConv(True,in_channels//2,in_channels,1)
        ])

    def forward(self,x):
        out = x
        for layer in self.block:
            out = layer(out)
        else:
            out = layer(out)
        y = out + x
        return y

class AutoregtessiveFlow(nn.Module):
    def __init__(self,in_shape,filters=64,kernel_size=7,n_layers=5,use_resblock=False,n_components=2):
        super().__init__()
        n_channels = shape[0]
        block_init = lambda: ResidualBlock(filters)

        model = nn.ModuleList([
            MaskConv(False,n_channels,filters,kernel_size=kernel_size,padding=kernel_size//2)
        ])

        for i in range(n_layers):
            model.append(LayerNorm(filters))
            model.extend([nn.ReLU(),block_init()])
            model.extend([nn.ReLU(),MaskConv(True,filters,filters,1)])
            model.extend([nn.ReLU(),MaskConv(True,filters,n_components*3*n_channels,1)])

        self.net = model
        self.input_shape = in_shape
        self.n_channels = n_channels
        self.n_components = n_components

    def forward(self,x):
        batch_size = x.shape[0]
        out = x.float()
        for layer in self.net:
            out = layer(out)
        return out.view(batch_size, 3* self.n_components,*self.input_shape)

    def nll(self,x):
        log, log_scale, logits = torch.chunk(self.forwad(x),3,dim=1)
        weights = F.softmax(logits,dim=1)
        log_det = Normal(log,log_scale.exp()).log_prob(x.unsqueeze(1).repeat(1,1,self.n_components,1,1))
        return -log_det

    def sample(self,z):
        samples = torch.zeros(z,*self.input_shape)
        with torch.no_grad():
            for i in range(self.input_shape[1]):
                for j in range(self.input_shape[2]):
                    for k in range(self.n_channels):
                        log, log_scale, logits = torch.chunk(self.forwad(samples),3,dim=1)
                        log, log_scale, logits = log[:,:,k,i,j], log_scale[:,:,k,i,j], logits[:,:,k,i,j]
                        probs = F.softmax(logits,dim=1)
                        centers = torch.multinomial(probs,1).unsqueeze(-1)
                        samples[:,k,i,j] = torch.normal(log[torch.arange(z),centers],log_scale[torch.arange(z),centers].exp())
        return samples.permute(0,2,3,1)
