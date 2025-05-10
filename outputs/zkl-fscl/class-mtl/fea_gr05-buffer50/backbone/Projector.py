#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/10/8 上午9:37

import torch.nn as nn
import torch.nn.functional as F

class Projector(nn.Module):

    def __init__(self, in_channels):
        super(Projector, self).__init__()

        self.fc1 = nn.Linear(in_channels, 256)
        self.fc2 = nn.Linear(256, in_channels)

    def forward(self, x):
        x1 = x.mean(-1) if len(x.shape) == 3 else x

        h = F.relu(self.fc1(x1))
        y = F.relu(self.fc2(h)) + x1

        # return x1

        return y