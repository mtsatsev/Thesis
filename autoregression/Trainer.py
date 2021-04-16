import typing
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def eval_loss(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in data_loader:
            x = x.to(ptu.device).float().contiguous()
            loss = model.nll(x)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss.item()


def dequantization(model,train_loader,optimizer):
    model.train()
    losses = []
    for x in train_loader:
        x = x.float().contiguous()
        x += torch.distributions.uniform(0.0,0.5).sample(x.shape)
        loss = model.nll(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def train(model,train_loader,test_loader,train_args):
    epochs = train_args['epochs']
    lr     = train_args['learning_rate']
    optimizer = optim.Adam(model.parameters,lr=lr)

    train_losses = []
    test_losses = []
    test_loss = eval_loss(model,test_loader)
    test_losses.append(test_loss)

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = dequantization(model,train_loader,optimizer)
        train_losses.extend(epoch_train_loss)

        test_loss = eval_loss(model,test_loader)
        test_losses.append(test_loss)

    return train_losses, test_losses
