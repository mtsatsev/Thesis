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
import time
import seaborn as sns
import pytorch_lightning as pl
from matplotlib.colors import to_rgb
from scipy import signal as ss
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from CouplingLayer import checkerBoardMask, CouplingLayer
from Flow import Flow
from rqs import unconstrained_rqs,rqs,searchsorted
from GatedResNet import ConcatELU, LayerNormChannels, GatedConv, GatedResNet

def train_flow(flow, model_name="MNISTFlow"):
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(default_root_dir=os.path.join(".",model_name),
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=200,
                         gradient_clip_val=1.0,
                         callbacks=[LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    train_data_loader = data.DataLoader(train_data, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)
    valid_data_loader = data.DataLoader(valid_data, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)
    result = None

    # Check whether pretrained model exists. If yes, load it and skip training
    print("Start training", model_name)
    trainer.fit(flow, train_data_loader, valid_data_loader)

    # Test best model on validation and test set if no result has been found
    # Testing can be expensive due to the importance sampling.
    if result is None:
        val_result = trainer.test(flow, test_dataloaders=valid_loader, verbose=not USE_NOTEBOOK)
        start_time = time.time()
        test_result = trainer.test(flow, test_dataloaders=test_loader, verbose=not USE_NOTEBOOK)
        duration = time.time() - start_time
        result = {"test": test_result, "val": val_result, "time": duration / len(test_loader) / flow.import_samples}

    return flow, result


def create_network(K=3):
    flow_layers = []
    flow_layers+=[CouplingLayer(network=GatedResNet(1,16,K*3-1),
                  mask=checkerBoardMask(h=28,w=28,inverse=(i%2==1)),
                  in_channels=1) for i in range(4)]

    flow_layers+=[CouplingLayer(network=GatedResNet(1,32,K*3-1),
                  mask=checkerBoardMask(h=28,w=28,inverse=(i%2==1)),
                  in_channels=1) for i in range(4)]

    prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
    shape = (128,28)
    flow_model = Flow(prior,flow_layers,shape)
    return flow_model

def discretize(example):
  return (example * 255).to(torch.int32)

if __name__ == '__main__':

    batch_size = 128
    '''
    train_data = torchvision.datasets.MNIST(root="~/Documents/nncourse",
                                       train=True,
                                       transform=transforms.ToTensor(), #.Compose([transforms.ToTensor(),#discretize]),
                                       download=True
                                      )

    valid_data = torchvision.datasets.MNIST(root="~/Documents/nncourse",
                                       train=False,
                                       transform=transforms.ToTensor(), #Compose([transforms.ToTensor(),discretize]),
                                       download=True
                                     )
    valid_data, test_data = torch.utils.data.random_split(valid_data, [6000, 4000])


    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


    print("The size of the training data is: {}. And the size of the testing data is: {} ".format(len(train_data),len(valid_data)))
    print("The size/shape of a single example is: {}.".format(train_data[0][0][0].size()))



    fig, ax = plt.subplots(2,6,figsize=(15,6))
    for i in range(ax.shape[1]):
        ax[0][i].imshow(train_data[i][0][0],cmap='gray')
        ax[0][i].set_title("From Training set")
        ax[1][i].imshow(valid_data[i][0][0],cmap='gray')
        ax[1][i].set_title("From Validation set")
    plt.show()
    '''
    x = torch.rand(128,1,28,28)
    nf = create_network()
    #x = discretize(torch.rand(1,28,28))
    #nf(x)
    nf(x)

    flow_dict = {'model':{}}

    flow_dict['model']['flow'],flow_dict['model']['result'] = train_flow(nf)
