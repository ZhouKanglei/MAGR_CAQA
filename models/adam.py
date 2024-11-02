# -*- coding: utf-8 -*-
# @Time: 2023/6/25 20:14

import torch
from models.utils.continual_model import ContinualModel
from utils.metrics import ListNetLoss


class Adam(ContinualModel):
    NAME = 'adam'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Adam, self).__init__(backbone, loss, args, transform)

        self.opt = torch.optim.Adam(
            params=[
                {'params': self.net.feature_extractor.parameters(),
                 'lr': self.args.lr},
                {'params': self.net.regressor.parameters(),
                 'lr': self.args.lr},
            ],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        self.graph_reg_loss = ListNetLoss()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, task=None):

        self.opt.zero_grad()

        outputs, features = self.net(inputs, 'all')
        loss = self.loss(outputs, labels)
        assert not torch.isnan(loss)
        loss.backward()

        self.opt.step()

        return loss.item()