import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class AUDIO_NORM(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features=num_features)

    def forward(self, x):
        return nn.BatchNorm2d.forward(
            self, x.transpose(1, 3),
        ).transpose(1, 3)

########################################################################
# torch AE model
########################################################################

class AEGMM(nn.Module):
    def __init__(
        self,
        n_gmm = 4,
        z_dim  = 8,
        latent_dim=10, # 2 loss+ middle hidden dims
        feature_dim=128,
        hidden_dims= [128, 128, 128, 8], # [128, 128, 128, 128, 8]
        enc_drop_out=0.2,
        dec_drop_out=0.2,
    ):
        """Network for AEGMM"""
        super(AEGMM, self).__init__()
        hidden_dims.insert(0, feature_dim)
        num_layer = len(hidden_dims) - 1

        # Encoder network
        layers = []
        for i in range(num_layer):
            layers.append(nn.Dropout(p=enc_drop_out, inplace=False))
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            #layers.append(nn.BatchNorm2d(1))
            layers.append(nn.ReLU(inplace=True))
        self.enc = nn.Sequential(*layers)

        # Decoder network
        layers = []
        for i in reversed(range(num_layer)):
            layers.append(nn.Dropout(p=dec_drop_out, inplace=False))
            layers.append(nn.Linear(hidden_dims[i+1], hidden_dims[i]))
            #layers.append(nn.BatchNorm2d(1))
            layers.append(nn.ReLU(inplace=True))
        self.dec = nn.Sequential(*layers)

        # Estimation network
        layers = []
        layers += [nn.Linear(latent_dim, 128)]
        layers += [nn.Tanh()]
        layers += [nn.Dropout(p=0.5)]
        layers += [nn.Linear(128, n_gmm)]
        layers += [nn.Softmax(dim=-1)]
        self.estimation = nn.Sequential(*layers)

        #self.register_buffer("phi", torch.zeros(n_gmm))
        #self.register_buffer("mu", torch.zeros(n_gmm, latent_dim))
        #self.register_buffer("cov", torch.zeros(n_gmm, latent_dim, latent_dim))
        # self.weight_init()

    def weight_init(self):
        for layers in [self.enc, self.dec]:
            for layer in layers:
                if type(layer) == nn.Linear:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    nn.init.normal_(layer.bias)

    def encoder (self, x):
        # x: (B, C, T, F)
        h  = self.enc(x)
        return h


    def decoder(self, h):
        y = self.dec(h)
        return y

    def estimate(self, z):
        gamma = self.estimation(z)
        return gamma

    def compute_reconstruction(self, x, x_hat):
        #size = x.size()
        #x_feature = x.view(size[0], size[2] * size[3])
        #x_hat_feature  = x_hat.view(size[0], size[2] * size[3])
        #relative_euclidean_distance = (x_feature-x_hat_feature).norm(2, dim=1) / x_feature.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=-1)
        relative_euclidean_distance = (x - x_hat).norm(2, dim=-1) / x.norm(2, dim=-1)
        return relative_euclidean_distance, cosine_similarity

    def forward(self, x):
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=-1)
        gamma = self.estimate(z)
        return z_c, x_hat, z, gamma