import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision


class MaskedConv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size ,mask_type, **kwargs):

        super(MaskedConv1d,self).__init__()

        if mask_type == 'A':
            self.mask = self.mask_of_type_A(kernel_size)
        else:
            self.mask = self.mask_of_type_B(kernel_size)
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = dilation*(kernel_size-1)//2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=padding, **kwargs)

    def forward(self,x):
        self.conv.weight.data *= self.mask
        return self.conv(x)

    def mask_of_type_A(self, kernel_size):
        mask = torch.ones(kernel_size)
        mask[kernel_size//2+1:] = 0
        return mask

    def mask_of_type_B(self, kernel_size):
        mask = torch.ones(kernel_size)
        mask[kernel_size//2:] = 0
        return mask
