import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

########################################################################
# torch AE model
########################################################################

class simple_autoencoder(nn.Module):
    def __init__(
        self,
        feature_dim=128,
        hidden_dims= [128, 128, 128, 128, 8],
        enc_drop_out=0.2,
        dec_drop_out=0.2,
    ):
        super().__init__()
        hidden_dims.insert(0, feature_dim)
        num_layer = len(hidden_dims) - 1
        layers = []
        for i in range(num_layer):
            layers.append(nn.Dropout(p=enc_drop_out, inplace=False))
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm2d(1))
            layers.append(nn.ReLU(inplace=True))
        self.enc = nn.Sequential(*layers)
        layers = []
        for i in reversed(range(num_layer)):
            layers.append(nn.Dropout(p=dec_drop_out, inplace=False))
            layers.append(nn.Linear(hidden_dims[i+1], hidden_dims[i]))
            layers.append(nn.BatchNorm2d(1))
            layers.append(nn.ReLU(inplace=True))
        self.dec = nn.Sequential(*layers)
        # self.weight_init()

    def weight_init(self):
        for layers in [self.enc, self.dec]:
            for layer in layers:
                if type(layer) == nn.Linear:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    nn.init.normal_(layer.bias)

    def forward(self, x):
        # x: (B, C, T, F)
        h = self.enc(x)
        y = self.dec(h)
        return y