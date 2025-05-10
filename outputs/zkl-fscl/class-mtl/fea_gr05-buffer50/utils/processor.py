# -*- coding: utf-8 -*-
# @Time: 2023/6/20 22:43
import os
import pprint

import math
import numpy as np
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from datasets import get_dataset
from models import get_model
from utils.loggers import Logger
from utils.misc import init_seed, count_param, mask_classes
from utils.status import ProgressBar

class Processor(object):
    def __init__(self, args):
        self.args = args
        # init seed
        self.init_seed()
        # load dataset
        self.load_dataset()
        # load model
        self.load_model()

    def init_seed(self):
        if self.args.seed is not None:
            self.args.seed += self.args.local_rank
            self.args.logging.info(f'Initialize seed with {self.args.seed}')
            init_seed(self.args.seed)

    def load_model(self):
        backbone = self.dataset.get_backbone()
        loss = self.dataset.get_loss()
        self.model = get_model(self.args, backbone, loss, self.dataset.get_transform())
        self.args.logging.info(f'Load model: {self.args.model} ('
                               f'{count_param(self.model.net.feature_extractor):,d} + '
                               f'{count_param(self.model.net.regressor):,d})')

        self.model = self.model.to(self.args.output_device)
        if self.args.local_rank != -1:
            self.model.net = DistributedDataParallel(self.model.net,
                                                     device_ids=[self.args.local_rank],
                                                     broadcast_buffers=False,
                                                     output_device=self.args.local_rank)
            self.model.module = self.model.net.module

        elif len(self.args.gpus):
            self.model.module = self.model.net
            self.model.net = DataParallel(self.model.net, device_ids=self.args.gpus)
        else:
            self.model.net = self.model.net.to(self.args.output_device)
            self.model.module = self.model.net

        # load pretrain model
        if self.args.pretrain:
            if os.path.exists(self.args.weight_path):
                state = torch.load(self.args.weight_path)
                self.model.load_state_dict(state['model'])
                self.args.logging.info(f'Load pretrained model state from {self.args.weight_path}.')
            else:
                self.args.logging.info(f'No pretrained model found at {self.args.weight_path}.')


    def load_dataset(self):
        self.dataset = get_dataset(args=self.args)
        self.args.logging.info(f'Load dataset: {self.args.dataset} ({self.dataset.SETTING})')

    def evaluate(self, dataset, last=False):
        status = self.model.net.training

        self.model.eval()
        self.accs, self.accs_mask_classes = [], []
        # transverse all test loaders before this task
        for k, test_loader in enumerate(dataset.test_loaders):
            if last and k < len(dataset.test_loaders) - 1: continue

            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            for data in test_loader:
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(self.args.output_device), \
                                     labels.to(self.args.output_device)
                    if 'class-il' not in self.model.COMPATIBILITY:
                        outputs = self.model(inputs, k)
                    else:
                        outputs = self.model(inputs)

                    _, pred = torch.max(outputs.data, 1)
                    correct += torch.sum((pred == labels).int()).item()
                    total += labels.shape[0]

                    if self.dataset.SETTING == 'class-il':
                        mask_classes(outputs, self.dataset, k)
                        _, pred = torch.max(outputs.data, 1)
                        correct_mask_classes += torch.sum((pred == labels).int()).item()

            self.accs.append(correct / total * 100
                             if 'class-il' in self.model.COMPATIBILITY else 0)
            self.accs_mask_classes.append(correct_mask_classes / total * 100)

        self.model.net.train(status)

        return self.accs, self.accs_mask_classes

    def train(self):
        # random model for fwt
        if self.model.NAME not in ['icarl', 'pnn']:

            dataset_copy = get_dataset(self.args)
            for t in range(self.dataset.N_TASKS):
                self.model.net.train()
                _, _ = dataset_copy.get_data_loaders()

            self.random_results_class, self.random_results_task = self.evaluate(dataset_copy)
        # trained model
        self.results, self.results_mask_classes = [], []
        logger = Logger(self.dataset.SETTING, self.dataset.NAME, self.model.NAME)
        progress_bar = ProgressBar(verbose=True)

        for t in range(self.dataset.N_TASKS):
            self.args.logging.info(f'| Task {t + 1:02d} |'.center(36, '-'))
            self.model.net.train()
            train_loader, test_loader = self.dataset.get_data_loaders()
            scheduler = self.dataset.get_scheduler(self.model, self.args)
            # beginning
            if hasattr(self.model, 'begin_task'):
                self.model.begin_task(self.dataset)
            # t >= 1
            if t:
                self.evaluate(self.dataset, last=True)
                self.results[t - 1] = self.results[t - 1] + self.accs
                if self.dataset.SETTING == 'class-il':
                    self.results_mask_classes[t - 1] = self.results_mask_classes[t - 1] + self.accs_mask_classes
            # sequential learning
            for epoch in range(self.args.n_epochs):
                # if joint training, skip the middle training
                if self.args.model == 'joint': continue
                # batch training
                for i, data in enumerate(train_loader):
                    # debug mode
                    if self.args.debug_mode and i > 3: break

                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(self.args.output_device), \
                                     labels.to(self.args.output_device)
                    not_aug_inputs = not_aug_inputs.to(self.args.output_device)
                    loss = self.model.meta_observe(inputs, labels, not_aug_inputs)

                    assert not math.isnan(loss)
                    progress_bar.prog(i, len(train_loader), epoch, t, loss)

                if scheduler is not None:
                    scheduler.step()

                # save model
                if self.args.local_rank < 1:
                    torch.save(self.model.state_dict(), self.args.weight_path)
                    self.args.logging.info(f'Save model state to {self.args.weight_path}.')

            if self.args.model != 'joint' and self.args.local_rank < 1: print()
            # ending task
            if hasattr(self.model, 'end_task'):
                self.model.end_task(self.dataset)
                if self.args.local_rank < 1 and t == self.dataset.N_TASKS - 1: print()
            # evaluation
            self.evaluate(self.dataset)
            # task level statistics
            self.task_stat(logger)

        # final statistics
        self.final_stat(logger)

    def task_stat(self, logger):
        self.results.append(self.accs)
        self.results_mask_classes.append(self.accs_mask_classes)

        mean_acc = np.mean([self.accs, self.accs_mask_classes], axis=1)
        logger.log(mean_acc)
        logger.log_fullacc(self.accs)

        if self.dataset.SETTING == 'domain-il':
            self.args.logging.info(f'Accuracy: {mean_acc[0]:.2f} %')
        else:
            self.args.logging.info(f'Accuracy [Class-IL]: {mean_acc[0]:.2f}%, '
                                   f'[Task-IL]: {mean_acc[1]:.2f}%')


    def final_stat(self, logger):
        # metric
        logger.add_bwt(self.results, self.results_mask_classes)
        logger.add_forgetting(self.results, self.results_mask_classes)
        if self.model.NAME not in ['icarl', 'pnn']:
            # calculation
            logger.add_fwt(self.results, self.random_results_class,
                           self.results_mask_classes, self.random_results_task)
        # log
        logger.write(vars(self.args))
        self.args.logging.info('Results: \n' + pprint.pformat(logger.dump()))

        # save model
        if self.args.local_rank < 1:
            torch.save(self.model.state_dict(), self.args.weight_path)
            self.args.logging.info(f'Save model state to {self.args.weight_path}.')

    def start(self):
        # train and test
        if self.args.phase == 'train':
            self.train()
        else:
            pass