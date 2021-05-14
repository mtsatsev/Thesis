import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
import matplotlib.pyplot as plt

from torch import distributions
from torch.utils.data import Dataset, DataLoader, random_split
from torch.distributions.multivariate_normal import MultivariateNormal
from inverseAutoregressiveFlow import InverseAutoregressiveFlow


class SineWave(Dataset):
    """ A sine wave whose parameters are sampled from distributions """

    def __init__(self, noise_strength: float):
        """ """

        # Setup the distributions
        self.amp_dist = distributions.Uniform(low=0.2, high=1)
        self.freq_dist = distributions.Uniform(low=0.1, high=0.25)
        self.phase_dist = distributions.Uniform(low=0, high=math.pi)
        self.noise_dist = distributions.Uniform(low=-noise_strength, high=noise_strength)

    def __len__(self):
        """ """
        return 100000

    def __getitem__(self, item):
        """ """
        amp = self.amp_dist.sample((1,))
        freq = self.freq_dist.sample((1,))
        phase = self.phase_dist.sample((1,))

        x = torch.linspace(start=-3 * math.pi, end=3 * math.pi, steps=24)

        return amp * torch.sin(2 * math.pi * freq * x + phase) + self.noise_dist.sample((24,)), torch.stack([amp, phase])

dataset = SineWave(noise_strength=0.1)

train_data, valid_data = random_split(dataset,[80000,20000])
valid_data, test_data  = random_split(valid_data,[10000,10000])

train_loader = DataLoader(dataset=train_data,batch_size=128,shuffle=True)
valid_loader = DataLoader(dataset=valid_data,batch_size=128,shuffle=True)
test_loader  = DataLoader(dataset=test_data, batch_size=128,shuffle=True)


def train(model,data_loader,epochs,optimizer):
    train_loss   = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for t,posterior in data_loader:
            optimizer.zero_grad()
            posterior = posterior.permute(0,2,1)
            t = t.reshape(128,1,-1)
            est_posterior,log_det = model(posterior,t)
            ll = MultivariateNormal(posterior,0.25*torch.eye(2))
            loss = -ll.log_prob(est_posterior) + log_det
            #print(log_det)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            print()
            print(epoch,loss.item())
            print("Posterior: {}".format(posterior[-1]))
            print("Estimated Posterior: {}".format(est_posterior[-1]))
            print(est_posterior.shape)
        print("Train epoch: {}, train_loss: {}".format(epoch,loss.item()))
        train_loss.append(np.mean(train_losses))
    return model, train_loss


def sample(model,data):
    for t,posterior in data:
        est_posterior = model.inverse(posterior,t)


IASF = InverseAutoregressiveFlow(1,3,T=3,K=15)

optimizer = optim.Adam(IASF.parameters(),lr=0.0001)

model, train_loss = train(IASF,train_loader,8,optimizer)

torch.save(model.state_dict(), "./IASP.pth")
