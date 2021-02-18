import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class ConcatELU(nn.Module):
    '''
    Activation function which applies ELU in both directions.
    '''
    def forward(self,x):
        return torch.cat([F.elu(x),F.elu(-x)],dim=1)

class LayerNormChannels(nn.Module):
    def __init__(self,in_channels):
        '''
        Applies normalization in accross the input channels.
        Inputs:
            Number of channels.
        '''
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self,x):
        x = x.permute(0,2,3,1)
        x = self.layer_norm(x)
        x = x.permute(0,3,1,2)
        return x

class GatedConv(nn.Module):
    def __init__(self,in_channels,hidden_channels):
        '''
        Create a two layer deep network for ResNet with input gate.
        Inputs:
            in_channels     - Number of input channels.
            hidden_channels - Number of hidden channels.
        '''
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels,kernel_size=3,padding=1),
            ConcatELU(),
            nn.Conv2d(2*hidden_channels, 2*in_channels, kernel_size=1)
        )
        self.l1 = nn.Conv2d(in_channels, hidden_channels,kernel_size=3,padding=1)
        self.lin = ConcatELU()
        self.l2 = nn.Conv2d(2*hidden_channels, 2*in_channels, kernel_size=1)

    def forward(self,x):
        out = self.net(x)
        val, gate = out.chunk(2,dim=1)
        return x + val * torch.sigmoid(gate)


class GatedResNet(nn.Module):
    def __init__(self,in_channels,hidden_channels,output_channels,num_layers=3):
        '''
        Creates GatedResNet using the previous modules.
        Inputs:
            in_channels     - Number of input channels.
            hidden_channels - Number of hidden channels.
            output_channels - Number of output channels (3K-1 * in_channels)
        '''
        super().__init__()
        layers = [nn.Conv2d(in_channels, hidden_channels,kernel_size=3,padding=1)]
        for _ in range(num_layers):
            layers += [
                GatedConv(hidden_channels,hidden_channels),
                LayerNormChannels(hidden_channels)
            ]
        layers += [
            ConcatELU(),
            nn.Conv2d(2*hidden_channels,output_channels,kernel_size=3,padding=1)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)


# Sanity check
K = 7
x = torch.rand(1,1,28,28)
res = GatedResNet(in_channels=1,hidden_channels=10,output_channels=3*K-1)
y = res(x)
print(y)
print(y.size())
W,H,D = torch.split(y,K,dim=1)
print(W.size())
print(H.size())
print(D.size())
