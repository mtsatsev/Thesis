import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from rqs import unconstrained_rqs
from pixelCNN import PixelCNN
class InverseAutoregressiveFlow(nn.Module):

    def __init__(self,c_in,c_out,T,K):
        super().__init__()
        self.network = nn.ModuleList()
        for t in range(T):
            self.network += [PixelCNN(c_in,c_out,n_layers=4,K=K,T=t)]
        self.T = T
        self.c_out = c_out

    def forward(self,z,c):
        #x = torch.from_numpy(np.random.normal(0,1,self.c_out))
        x = torch.from_numpy(np.random.normal(0,1,z.shape)).to(torch.float32)

        log_det = torch.zeros(z.shape[0],1)

        for t in range(self.T):
            log_temp = torch.zeros_like(log_det)
            W,H,D = self.network[t](c,x)
            #torch.Size([128, 1, 10])
            #torch.Size([128, 1, 10])
            #torch.Size([128, 1, 9])
            for i in range(x.shape[-1]):
                x[:,:,i], ld = unconstrained_rqs(x[:,:,i],W, H, D,inverse=True)
                log_temp += ld
            log_temp = log_temp/x.shape[-1]
            log_det += log_temp
        return x, log_det

    def inverse(self,x,c):
        z = torch.from_numpy(np.random.normal(0,1,x.shape)).to(torch.float32)
        log_det = torch.zeros(x.shape[0],1)

        for t in range(self.T):
            log_temp = torch.zeros_like(log_det)
            W,H,D = self.network[t](c,z)
            '''
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print("THE TIME POINT AND WHICH IT FAILED IS {}".format(t))
            print("The shape of x is {}".format(x.shape))
            print("The shape of z is {}".format(z.shape))
            print("The shape of W is {}".format(W.shape))
            print("The shape of D is {}".format(D.shape))
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            '''
            for i in range(z.shape[-1]):
                z[:,:, i], ld = unconstrained_rqs(z[:,:,i],W, H, D,inverse=False)
                log_temp += ld
            log_temp = log_temp/z.shape[-1]
            log_det += log_temp
        return z, log_det

    def sample(self,n_samples,z):
        samples = []
        for _ in n_samples:
            x,_ = self.forward(z)
            samples.appned(x)
        return samples

'''
IAF = InverseAutoregressiveFlow(1,3,24)
h = torch.rand(128,1,24)
x = torch.randn(128,1,3)
z,ld = IAF(x,h)
print(ld.shape)


count = 0
for p in IAF.parameters():
    count +=1

print(count)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

print(get_n_params(IAF))

model_parameters = filter(lambda p: p.requires_grad, IAF.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
'''
