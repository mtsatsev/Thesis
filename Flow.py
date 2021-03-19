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

class Flow(pl.LightningModule):

    def __init__(self,prior,flows,shape):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.prior = prior
        self.shape = shape

    def forward(self,x):
        return self.get_likelyhood(x)

    def encode(self,x):
        z, log_det = x, torch.zeros_like(x)
        print("encode")
        print(z.size())
        print("encode")

        for flow in self.flows:
            z, log_det = flow(z,log_det,reverse=False)
        return z, log_det

    def get_likelyhood(self,x):
        z, log_det = self.encode(x)
        log_pz = self.prior.log_prob(z)

        print(log_det.size())
        print(log_pz.size())
        log_px = log_det + log_pz
        return log_px

    @torch.no_grad()
    def sample(self,n_samples):
        z = self.prior.sample(sample_shape=self.shape)
        log_det = torch.zeros_like(z)
        for flow in self.flows[::-1]:
            z,log_det = flow(z,log_det,reverse=True)
        return z

    def training_step(self,batch,batch_idx):
        loss = self.get_likelyhood(batch[0])
        self.log('train_bpd',loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # An scheduler is optional, but can help in flows to get the last bpd improvement
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def validation_step(self,batch,batch_idx):
        loss = self.get_likelyhood(batch[0])
        self.log('val_bpd',loss)

    def testing_step(self,batch,batch_idx):
        samples = []
        for i in range(6):
            x_ll = self.get_likelyhood(batch[0])
            samples.append(x_ll)
        x_ll = torch.stack(samples,dim=-1)

        ''' Go from log-space to exp-space and back to log-space '''
        x_ll = torch.logsumexp(x_ll,dim=-1) - np.log(self.n_samples)

        ''' Calculate the final bpd '''
        bpd = -x_ll * np.log2(np.exp(1) / np.prod(batch[0].shape[1:]))
        bpd = bpd.mean()
        self.log('test_bpd',bpd)
