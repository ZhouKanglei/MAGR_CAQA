# -*- coding: utf-8 -*-
# @Time: 2023/6/24 20:37
import os
import pprint

import math
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from datasets import get_dataset
from utils.loggers import Logger
from utils.processor import Processor
from utils.status import ProgressBar
from utils.misc import distributed_concat, fix_bn

class ProcessorAQA(Processor):

    def __init__(self, args):
        super(ProcessorAQA, self).__init__(args)
        self.args = args

        self.current_epoch = -1
        self.current_task = -1
        self.best_epoch = -1
        self.best_rho = 0
        self.best_metrics = {'rho': 0, 'p': -1, 'L2': 1e10, 'RL2': 1e10}

        self.best_weight_path = os.path.join(os.path.dirname(self.args.weight_save_path), 'best.pth')
        self.last_weight_path = os.path.join(os.path.dirname(self.args.weight_save_path), 'last.pth')

    def compute_metric(self, pred_scores, true_scores):
        rho, p = stats.mstats.spearmanr(pred_scores, true_scores)
        rho *= 100

        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)

        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = 100 * np.power((pred_scores - true_scores) /
                             (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]

        self.current_metrics = {'rho': rho, 'p': p, 'L2': L2, 'RL2': RL2}
        if self.best_metrics['rho'] < rho:
            self.best_epoch = self.current_epoch
            self.best_metrics = {'rho': rho, 'p': p, 'L2': L2, 'RL2': RL2}
            # if self.args.phase == 'train': self.save_best()

        return rho, p, L2, RL2

    def load_best_model(self, weight_path=None):
        if weight_path is None and self.args.phase == 'train': return
        elif weight_path is None:
            weight_path = self.args.weight_save_path.replace('weight.pth', 'best.pth')

        if os.path.exists(weight_path):
            state = torch.load(weight_path)
            self.model.load_state_dict(state['model'], strict=False)
            self.args.logging.info(f'Load pretrained model state from {weight_path}.')
            self.args.logging.info('The current metrics: \n' + pprint.pformat(state['current_metrics']))
            # self.args.logging.info('Best: \n' + pprint.pformat(state['best_metrics']))
        else:
            self.args.logging.info(f'No pretrained model found at {weight_path}.')
            weight_path = weight_path.replace('best.pth', 'last.pth')

            if os.path.exists(weight_path):
                state = torch.load(weight_path)
                self.model.load_state_dict(state['model'], strict=False)
                self.args.logging.info(f'Load pretrained model state from {weight_path}.')
                self.args.logging.info('Current: \n' + pprint.pformat(state['current_metrics']))
                self.args.logging.info('Best: \n' + pprint.pformat(state['best_metrics']))

    def save_checkpoint(self, epoch, metrics, filename='last'):

        checkpoint_file = self.best_weight_path if filename == 'best' else self.last_weight_path
        if filename != 'best' and filename != 'last':
            checkpoint_file = checkpoint_file.replace('best.pth', f'{filename}.pth')
            checkpoint_file = checkpoint_file.replace('last.pth', f'{filename}.pth')

        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

        torch.save({
            'model': self.model.state_dict(),
            'epoch': epoch, 'task': self.current_task,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'current_metrics': metrics
        }, checkpoint_file)

        if filename == 'best':
            self.args.logging.info(f'Save checkpoint to {checkpoint_file}.')

    def save_best(self):
        if self.args.local_rank < 1:
            self.args.logging.info(f'New best found (e{self.current_epoch}):\n' +
                                   pprint.pformat(self.best_metrics))
            self.save_checkpoint(self.best_epoch, self.best_metrics, 'best')

    def evaluate(self, dataset, last=False):
        status = self.model.net.training
        self.model.net.eval()

        self.rhos, self.rhos_mask_classes = [], []
        # transverse all test loaders before this task
        if 'joint' in self.model.NAME: dataset.test_loaders = [dataset.get_joint_loader('test')]

        all_label_scores, all_output_scores = [], []
        for k, test_loader in enumerate(dataset.test_loaders):
            # last
            if last and k < len(dataset.test_loaders) - 1: continue

            label_scores, output_scores = [], []

            for data in test_loader:
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(self.args.output_device), \
                                     labels.to(self.args.output_device)

                    outputs = self.model(inputs)

                    if labels.shape != outputs.shape:
                        outputs = outputs * labels[:, 1:]
                        labels = labels[:, :1]

                    if self.args.local_rank != -1:
                        labels = distributed_concat(labels)
                        outputs = distributed_concat(outputs)

                    label_scores.extend(labels.cpu().numpy().reshape(-1,).tolist())
                    output_scores.extend(outputs.cpu().numpy().reshape(-1,).tolist())

            rho, p, L2, RL2 = self.compute_metric(output_scores, label_scores)

            self.rhos.append(rho)
            self.rhos_mask_classes.append(rho * 0)

            self.args.logging.info(f'Evaluate T{k + 1:02d}, {len(output_scores):3d} samples, '
                                   f'rho:{rho:6.2f} %, p: {p:5.4f}, L2: {L2:7.2f}, RL2: {RL2:8.2f}')

            all_label_scores.extend(label_scores)
            all_output_scores.extend(output_scores)

            self.all_label_scores, self.all_output_scores = all_label_scores, all_output_scores

            if set(all_label_scores) != set(label_scores):
                rho, p, L2, RL2 = self.compute_metric(all_output_scores, all_label_scores)
                self.args.logging.info(f'      T01-{k + 1:02d}, {len(all_output_scores):3d} samples, '
                                       f'rho:{rho:6.2f} %, p: {p:5.4f}, L2: {L2:7.2f}, RL2: {RL2:8.2f}')
            self.overall_metrics = {'rho': rho, 'p': p, 'L2': L2, 'RL2': RL2}

        self.model.net.train(status)

        return self.rhos, self.rhos_mask_classes

    def training_loop(self, train_loader, progress_bar, scheduler, t):
        # sequential learning
        self.model.net.apply(fix_bn)

        for epoch in range(self.args.n_epochs):
            if self.args.local_rank is not -1: train_loader.sampler.set_epoch(epoch)
            self.current_epoch = epoch
            # batch training
            for i, data in enumerate(train_loader):
                # debug mode
                if self.args.debug_mode and i > 3: break

                if hasattr(train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(self.args.output_device)
                    labels = labels.to(self.args.output_device)
                    not_aug_inputs = not_aug_inputs.to(self.args.output_device)
                    logits = logits.to(self.args.output_device)
                    loss = self.model.meta_observe(inputs, labels, not_aug_inputs, epoch, t, logits=logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(self.args.output_device), \
                                     labels.to(self.args.output_device)
                    not_aug_inputs = not_aug_inputs.to(self.args.output_device)
                    loss = self.model.meta_observe(inputs, labels, not_aug_inputs, epoch, t)

                # assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

            if scheduler is not None:
                scheduler.step()

            if 'joint' in self.model.NAME:
                self.evaluate(self.dataset)

        if self.args.local_rank < 1: print()

    def train(self):
        # random model for fwt
        if 'joint' not in self.model.NAME:
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
            # initializing
            self.args.logging.info(f'| Task {t + 1:02d} |'.center(36, '-'))

            self.current_task = t
            self.best_epoch = -1
            self.best_rho = 0
            self.best_metrics = {'rho': 0, 'p': -1, 'L2': 1e10, 'RL2': 1e10}

            self.model.net.train()
            train_loader, test_loader = self.dataset.get_data_loaders()
            scheduler = self.dataset.get_scheduler(self.model, self.args)
            # beginning task
            if hasattr(self.model, 'begin_task'):
                self.model.begin_task(self.dataset)
            # t >= 1
            if t and self.dataset.SETTING != 'domain-il' and 'joint' not in self.model.NAME:
                self.evaluate(self.dataset, last=True)
                self.results[t - 1] = self.results[t - 1] + self.rhos
                if self.dataset.SETTING == 'class-il':
                    self.results_mask_classes[t - 1] = self.results_mask_classes[t - 1] + self.rhos_mask_classes

            # middle task
            if os.path.exists(self.args.base_pretrain_model_path) and t == 0 \
                    and self.args.base_pretrain and 'joint' not in self.model.NAME:
                self.load_best_model(weight_path=self.args.base_pretrain_model_path)
            elif 'lp_joint' in self.model.NAME:
                self.training_loop(train_loader, progress_bar, scheduler, t)
            elif 'joint' in self.model.NAME and t < self.dataset.N_TASKS - 1:
                continue
            elif 'joint' not in self.model.NAME:
                self.training_loop(train_loader, progress_bar, scheduler, t)

            # ending task
            if hasattr(self.model, 'end_task'):
                self.model.end_task(self.dataset)
                # if self.args.local_rank < 1 and t == self.dataset.N_TASKS - 1: print()
            # evaluation
            self.evaluate(self.dataset)
            # task level statistics
            self.task_stat(logger)

        # final statistics
        self.final_stat(logger)

    def task_stat(self, logger):
        self.results.append(self.rhos)
        self.results_mask_classes.append(self.rhos_mask_classes)

        mean_rho = np.mean([self.rhos, self.rhos_mask_classes], axis=1)
        logger.log(mean_rho)
        logger.log_fullacc((self.rhos, self.rhos_mask_classes))

        self.args.logging.info(f'Rho: {mean_rho[0]:.2f} %')
        self.args.logging.info(f'Rho (overall): {self.overall_metrics["rho"]:.2f} %')

        self.save_checkpoint(-1, self.current_metrics,
                             filename=f'last-task{self.current_task}')

    def final_stat(self, logger):
        # metric
        if self.dataset.SETTING != 'domain-il' and 'joint' not in self.model.NAME:
            logger.add_bwt(self.results, self.results_mask_classes)
            logger.add_forgetting(self.results, self.results_mask_classes)

            if self.model.NAME not in ['icarl', 'pnn']:
                # calculation
                logger.add_fwt(self.results, self.random_results_class,
                               self.results_mask_classes, self.random_results_task)
        # log
        logger.write(vars(self.args))
        self.args.logging.info('Results: \n' + pprint.pformat(logger.dump()))

    def test(self):
        for t in range(self.dataset.N_TASKS):
            self.model.net.eval()
            _, _ = self.dataset.get_data_loaders()

        self.random_results_class, self.random_results_task = self.evaluate(self.dataset)

    def start(self):
        # train and test
        if self.args.phase == 'train':
            if self.model.NAME == 'lp_joint':
                self.load_best_model(weight_path=self.args.weight_path)

            self.train()
        else:
            self.load_best_model(weight_path=self.args.weight_path)
            self.test()