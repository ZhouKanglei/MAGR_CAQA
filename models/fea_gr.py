#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/10/4 上午10:02

import copy
import torch
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer
from utils.misc import distributed_concat
from utils.metrics import ListNetLoss

from torch.nn.parallel import DistributedDataParallel


class FeaGr(ContinualModel):
    # feature replay with graph regularization
    NAME = 'fea_gr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(FeaGr, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, 'cpu')
        self.buffer.empty()

        self.current_task = 0
        self.i = 0
        self.lambda1 = args.alpha
        self.lambda2 = args.beta
        self.n_tasks = args.n_tasks + 1 if args.fewshot else args.n_tasks

        self.opt = torch.optim.Adam(
            params=[
                {'params': self.net.feature_extractor.parameters(),
                 'lr': self.args.lr},
                {'params': self.net.projector.parameters(),
                 'lr': self.args.lr},
                {'params': self.net.regressor.parameters(),
                 'lr': self.args.lr}
            ],
            lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        self.graph_reg_loss = ListNetLoss()

    def select_examples(self, lab, num):

        if len(lab) <= num:
            return list(range(len(lab)))
        else:
            selected_idxes = []

        sample_interval = len(lab) / num
        sample_ids = [int(i * sample_interval) for i in range(num)]

        scores = lab
        if len(lab.shape) == 2:
            if lab.shape[1] == 2:
                scores = lab[:, 0]
        scores = scores.reshape(-1, )
        sorted_ids = sorted(range(len(scores)), key=lambda k: scores[k])

        for i, sorted_id in enumerate(sorted_ids):
            if sorted_id in sample_ids:
                selected_idxes.append(i)

        return selected_idxes

    def fea2buffer_ous(self, dataset):
        # statistic
        examples_per_task = self.args.buffer_size // (self.n_tasks - 1)
        self.args.logging.info(
            f'Current task {self.current_task} - select {examples_per_task} samples'
        )

        # gather all labels and globally select examples
        all_labels = []
        for i, data in enumerate(dataset.train_loader):
            _, labels, _ = data

            if labels.shape[1] == 2:
                labels_ = list(labels[:, 0].numpy())
            else:
                labels_ = list(labels.reshape(labels.shape[0], ).numpy())
            all_labels.extend(labels_)

        all_labels_tensor = torch.Tensor(all_labels).to(self.device)
        all_labels = distributed_concat(all_labels_tensor).reshape(-1, 1)

        select_num = examples_per_task
        selected_idxes = self.select_examples(all_labels, select_num)
        selected_labels = all_labels[selected_idxes]
        selected_labels = list(selected_labels.cpu().numpy().reshape(-1, ))

        # current task
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(dataset.train_loader):
                inputs, labels, not_aug_inputs = data
                # select examples in a batch
                if labels.shape[1] == 2:
                    labels_ = list(labels[:, 0].numpy())
                else:
                    labels_ = list(labels.reshape(labels.shape[0], ).numpy())

                labels_tensor = torch.Tensor(labels_).to(self.device)
                labels_ = distributed_concat(labels_tensor)

                selected_idxes = []
                for j, label in enumerate(labels_):
                    if label in selected_labels:
                        selected_idxes.append(j)
                        selected_labels.pop(selected_labels.index(label))

                selected_num = len(selected_idxes)
                if selected_num == 0: continue

                selected_idxes = torch.Tensor(selected_idxes).to(self.device).long()
                # forward
                inputs = inputs.to(self.device)
                features = self.module.feature_extractor(inputs)
                # outputs = self.module.regressor(features)
                labels = labels.to(self.device)
                # gather features
                features = distributed_concat(features)[selected_idxes]
                labels = distributed_concat(labels)[selected_idxes]
                # add data
                self.buffer.add_data(
                    examples=features.data.cpu(),
                    logits=labels.data.cpu(),
                    task_labels=torch.ones(selected_num) * (self.current_task)
                )

        # statistic
        buf_x, buf_lab, buf_tl = self.buffer.get_all_data()
        for ttl in buf_tl.unique():
            idx = (buf_tl == ttl)
            if ttl > 0:
                self.args.logging.info(f"Task {int(ttl)} has {sum(idx)} samples in the buffer.")

    def fea2buffer(self, dataset):
        examples_per_task = self.args.buffer_size // (self.n_tasks - 1)
        self.args.logging.info(f'Current task {self.current_task} - {examples_per_task}')

        # current task
        counter = 0
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(dataset.train_loader):

                if examples_per_task - counter > 0:
                    inputs, labels, not_aug_inputs = data

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    features = self.module.feature_extractor(inputs)
                    outputs = self.module.regressor(features)

                    features = distributed_concat(features)
                    labels = distributed_concat(labels)

                    batch_size = outputs.shape[0]
                    if (examples_per_task - counter) // batch_size:
                        num_select = batch_size
                    else:
                        num_select = examples_per_task - counter

                    self.buffer.add_data(
                        examples=features.data.cpu()[:num_select],
                        logits=labels.data.cpu()[:num_select],
                        task_labels=torch.ones(num_select) * (self.current_task)
                    )

                counter += outputs.shape[0]

        # statistics
        buf_x, buf_lab, buf_tl = self.buffer.get_all_data()
        for ttl in buf_tl.unique():
            idx = (buf_tl == ttl)

            self.args.logging.info(f"Task {int(ttl)} has {sum(idx)} samples in the buffer.")

    def update_buffer(self):
        buf_x, buf_lab, buf_tl = self.buffer.get_data(
            self.buffer.num_seen_examples, transform=self.transform)

        with torch.no_grad():
            buf_x = buf_x.to(self.device)
            buf_x_tilde = self.net.module.projector(buf_x)

        self.buffer.empty()

        self.buffer.add_data(
            examples=buf_x_tilde.data.cpu(),
            logits=buf_lab,
            task_labels=buf_tl
        )

    def slow_learning(self):
        self.opt = torch.optim.Adam(
            params=[
                {'params': self.module.feature_extractor.parameters(),
                 'lr': self.args.lr * 0.01},
                {'params': self.module.projector.parameters(),
                 'lr': self.args.lr},
                {'params': self.module.regressor.parameters(),
                 'lr': self.args.lr}
            ],
            lr=self.args.lr, weight_decay=self.args.weight_decay
        )
    
    def end_task(self, dataset):
        self.current_task += 1
        # after the first epoch then fix some layers or reduce the learning rate
        if self.current_task == 1:
            self.args.logging.info('Learning slowly ...')
            self.slow_learning()
        # update memory buffer
        if not self.buffer.is_empty():
            self.update_buffer()
        # add features to buffer for replay
        if self.current_task < self.n_tasks:
            self.fea2buffer_ous(dataset)
        # copy the previous model for training the projector
        self.old_feature_extractor = copy.deepcopy(self.module.feature_extractor)

    def observe(self, inputs, labels, not_aug_inputs, epoch=True, task=True):

        self.i += 1
        self.opt.zero_grad()
        # incremental data
        outputs, features = self.net(inputs, 'all')
        loss_d_score = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            # feature projector learning
            with torch.no_grad():
                self.old_feature_extractor.eval()
                old_features = self.old_feature_extractor(inputs)
            features_hat = self.net.module.projector(old_features.data)
            loss_p_fea = F.mse_loss(features_hat, features.data)

            #  replay
            buf_feas, buf_logits, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_feas, buf_logits = buf_feas.to(self.device), buf_logits.to(self.device)

            # joint graph reconstruction
            buf_feas_hat = self.net.module.projector(buf_feas)
            joint_features = torch.cat([buf_feas_hat, features], dim=0)
            joint_labels = torch.cat([buf_logits, labels], dim=0)
            loss_j_reg = self.graph_reg_loss(joint_features, joint_labels,
                                             blocking=self.args.minibatch_size)

            # regressor alignment
            buf_outputs = self.net.module.regressor(buf_feas_hat)
            loss_m_score = self.loss(buf_outputs, buf_logits)

            # loss
            loss = loss_d_score + loss_m_score + \
                   self.lambda1 * loss_p_fea + self.lambda2 * loss_j_reg

        else:
            loss_d_reg = self.graph_reg_loss(outputs, labels)
            loss = loss_d_score + loss_d_reg

        assert not torch.isnan(loss)
        loss.backward()
        self.opt.step()

        return loss.item()
