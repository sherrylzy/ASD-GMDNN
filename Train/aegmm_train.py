import torch
from torch import optim
import  torch.nn.functional as F

import numpy as np
from barbar import Bar
from ASD.utils.utils import weights_init_normal
from ASD.Module.AEGMM import AEGMM
from ASD.Train.forward_step import ComputeLoss
from ASD.Preprocess import common as com
import torch.nn as nn

class TrainerAEGMM:
    """Trained class for AEGMM"""
    def __init__(self, param , data, device):
        self.param = param
        self.train_load = data
        self.device = device

    def train(self):
        """training the AEGMM model"""

        self.model= AEGMM(self.param["n_gmm"], self.param["latent_dim"]).to(self.device)
        self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.param["lr"])
        self.compute = ComputeLoss(self.model, self.param["lambda_energy"], self.param["lambda_cov"],
                                   self.device, self.param["n_gmm"])

        self.model.train()
        length = len(self.train_load.dataset)
        for epoch in range(self.param["num_epochs"]):
            total_loss = 0
            for x in Bar(self.train_load):
                x = x.float().to(self.device) #x: torch tensor
                optimizer.zero_grad()

                _, x_hat, z, gamma = self.model(x)

                loss = self.compute.forward(x, x_hat, z, gamma)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5) #Clips gradient norm of an iterable of parameters.
                optimizer.step()

                total_loss += loss.item()/(1e+11)
            print('Training AEGMM... Epoch: {}, Loss: {:.3f}'.format(
                epoch, total_loss/len(self.train_load.dataset)
            ))

