# -*- coding: utf-8 -*-
# @Time: 2023/6/23 17:07
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import misc


class DAE(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DAE, self).__init__()

        self.fc1 = nn.Linear(in_channels, 256)
        self.fch = nn.Linear(256, 128)
        self.fc2_mean = nn.Linear(128, out_channels)
        self.fc2_logvar = nn.Linear(128, out_channels)

    def encode(self, x):
        h0 = F.relu(self.fc1(x))
        h1 = F.relu(self.fch(h0))
        mu = self.fc2_mean(h1)
        logvar = self.fc2_logvar(h1)
        return mu, logvar

    def reparametrization(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        esp = torch.randn(*mu.size()).to(mu.device) 
        
        z = mu + std * esp 
        return z

    def forward(self, x):
        h = x.mean(-1) if len(x.shape) == 3 else x

        mu, logvar = self.encode(h)
        z = self.reparametrization(mu, logvar)

        return z


if __name__ == '__main__':
    x = torch.randn((1, 1024, 10))
    model = DAE(1024, 1)
    y = model(x)
    print(f'{misc.count_param(model):,d}')

    from thop import profile

    flops, params = profile(model, inputs=(x.to('cpu'),))
    print(f'{flops / 1e9:.4f} GFLOPs')
