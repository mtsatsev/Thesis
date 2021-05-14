import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torch.nn.init as init

from maskedConvolution import MaskedConv1d
from GatedResNet import GatedResNet

class PixelCNN(nn.Module):
        def __init__(self, c_in, c_out ,n_layers=5, K=3, B=8 ,kernel_size=7,T=24):
        super(PixelCNN,self).__init__()
        self.c_out = c_out

        ''' PixelCNN part '''
        self.init_layer = MaskedConv1d(c_in,c_out, kernel_size,mask_type='A')
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers += nn.ModuleList([MaskedConv1d(c_out,c_out, kernel_size//2,mask_type='B')])
            self.layers += nn.ModuleList([nn.Sigmoid()])


        ''' Splines part '''
        self.init_splines = nn.Parameter(torch.Tensor(3 * K - 1))
        self.K = K
        self.B = B
        self.network = GatedResNet(c_out,c_out,3 * K - 1)
        #self.linear = nn.Linear(T,1)
        #torch.Size([128, 1, 1])
        #[128,1,3]  []
        ## Make rational quadratic splines
    def forward(self,x,prior):
        x = self.init_layer(x)
        for layer in self.layers:
            x = layer(x)
        return self.create_splines(x,prior)

    def create_splines(self,x,prior):
        prior = torch.cat(self.c_out*[prior],dim=1)
        x = torch.cat((x,prior),dim=2)
        out = self.network(x)
        W,H,D = torch.split(out,self.K,dim=-1)
        W = torch.softmax(W, dim=-1)
        H = torch.softmax(H, dim=-1)

        W = W * 2 * self.B
        H = H * 2 * self.B

        D = F.softplus(D)
        return W,H,D

    def get_init_splines(self,x):
        init_splines = self.init_splines.expand(x.shape[0],1,3 * self.K - 1)
        W, H, D = torch.split(init_splines, self.K, dim=-1)

        W = torch.softmax(W, dim=-1)
        H = torch.softmax(H, dim=-1)
        W = W * 2 * self.B
        H = H * 2 * self.B

        D = F.softplus(D)
        return W, H, D

    def reset_init_splines(self):
        init.uniform_(self.init_splines, -1/2, 1/2)

    def nll(self,x):
        pred = self.forward(x)
        x = x.reshape(128,-1)
        x = x.long()
        nll = F.cross_entropy(pred,x,reduction='none')
        bpd = nll.mean()
        return bpd.mean()

    @torch.no_grad()
    def sample(self,z_shape):
        x = torch.zeros((z))
        pred = self.forward(x)
        probs = F.softmax(pred,dim=-1)
        sample = torch.multinomial(probs,num_samples=1)
        return sample


'''
p = PixelCNN(1,1,K=10)
x = torch.rand(128,1,24)
W,H,D = p(x)
print(W.shape)
print(H.shape)
print(D.shape)
'''
