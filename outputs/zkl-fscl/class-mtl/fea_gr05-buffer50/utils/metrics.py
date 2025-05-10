# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import rankdata

def backward_transfer(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)


def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = []
    for i in range(1, n_tasks):
        li.append(results[i - 1][i] - random_results[i])

    return np.mean(li)


def forgetting(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)


class TripletCircleLoss(nn.Module):

    def __init__(self, margin=0.3, batch_size=128, view_num=3):
        super(TripletCircleLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def compute_loss(self, inputs, targets):
        idx_list = [(0, 1), (0, 2), (0, 2)]
        # Compute ranks
        batch_size = len(inputs)
        diff = torch.abs(targets - targets.t())
        diff = torch.Tensor([diff[idx_list[0]], diff[idx_list[1]], diff[idx_list[2]]])
        diff = diff.to(inputs.device)
        _, idx = torch.sort(diff)
        # Compute distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size)
        dist = dist + dist.t()
        dist = dist - 2 * torch.matmul(inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # Combination
        dist_ap = torch.Tensor([dist[idx_list[idx[0]]], dist[idx_list[idx[1]]]])
        dist_ap = dist_ap.to(inputs.device)
        dist_an = torch.Tensor([dist[idx_list[idx[1]]], dist[idx_list[idx[2]]]])
        dist_an = dist_an.to(inputs.device)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss

    def forward(self, inputs, targets):
        batch_size = len(inputs)
        if batch_size < 3: return 0

        loss = None
        for i in range(batch_size - 2):
            loss_ = self.compute_loss(inputs[i:i+3], targets[i:i+3])
            loss = loss + loss_ if loss is not None else loss_

        loss = loss / (batch_size - 2)

        return loss


class ListNetLoss(nn.Module):
    def __init__(self, eps=1e-7, padded_value_indicator=-1):
        super(ListNetLoss, self).__init__()

        self.eps = eps
        self.padded_value_indicator = padded_value_indicator

    def minmax(self, inputs):
        min_value, _ = inputs.min(dim=-1, keepdim=True)
        max_value, _ = inputs.max(dim=-1, keepdim=True)

        return (inputs - min_value + self.eps/2) / (max_value - min_value + self.eps)

    def listNet(self, y_pred, y_true, kl=True):
        """
        ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :return: loss value, a torch.Tensor
        """
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        pred_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        # pred_smax = self.minmax(y_pred)
        # true_smax = self.minmax(y_true)

        if kl:
            jsd = 0.5 * self.kld(pred_smax, true_smax) + 0.5 * self.kld(true_smax, pred_smax)
            return jsd

        pred_smax = pred_smax + self.eps
        pred_log = torch.log(pred_smax)

        return torch.mean(-torch.sum(true_smax * pred_log, dim=1))

    def kld(self, p, q):
        # p : batch x n_items
        # q : batch x n_items
        return (p * torch.log2(p / q + self.eps)).sum()

    def euclidean_dist(self, inputs):
        batch_size = inputs.shape[0]
        # Euclidean distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size)
        dist = dist + dist.t()
        dist = dist - 2 * torch.matmul(inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        return dist

    def angular_dist(self, inputs):
        # Angle distance
        # normalize the vectors
        n_inputs = inputs / (torch.norm(inputs, dim=-1, keepdim=True))
        # compute cosine of the angles using dot product
        cos_ij = torch.einsum('bm, cm->bc', n_inputs, n_inputs)
        cos_ij = cos_ij.clamp(min=-1 + self.eps, max=1 - self.eps)
        ang_dist = torch.acos(cos_ij)

        return ang_dist

    def forward(self, inputs_, targets_, blocking=None, wo_iig=False, wo_jg=False):
        # transformation
        if len(inputs_.shape) == 3:
            inputs = inputs_.mean(-1)
        else:
            inputs = inputs_

        if targets_.shape[1] == 2:
            targets = targets_[:, :1]
        else:
            targets = targets_

        # prediction
        dist = self.angular_dist(inputs)
        # gt
        diff = torch.abs(targets - targets.t())
        diff_rank = diff.argsort(descending=False).argsort(descending=False).to(torch.float32)

        # block
        if blocking == None:
            loss = self.listNet(dist, diff)
        else:
            b1 = blocking
            loss_joint = self.listNet(dist, diff)

            loss11 = self.listNet(dist[:b1, :b1], diff[:b1, :b1])
            loss12 = self.listNet(dist[:b1, b1:], diff[:b1, b1:])
            loss21 = self.listNet(dist[b1:, :b1], diff[b1:, :b1])
            loss22 = self.listNet(dist[b1:, b1:], diff[b1:, b1:])

            loss = loss_joint + loss11 + loss12 + loss21 + loss22

            if wo_iig:
                loss = loss_joint
            if wo_jg:
                loss = loss11 + loss12 + loss21 + loss22

        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist = dist - 2 * torch.matmul(inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class DRLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 reg_lambda=0.
                 ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(
            self,
            feat,
            target,
            h_norm2=None,
            m_norm2=None,
            avg_factor=None,
    ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)

        loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)

        return loss * self.loss_weight
