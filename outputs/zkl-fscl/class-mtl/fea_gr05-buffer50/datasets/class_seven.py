#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/10/21 下午12:02

import glob
import os
import pickle
import random
import math

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvideotransforms import video_transforms, volume_transforms
from backbone.AQAMLP import AQAMLP
from datasets.utils.continual_dataset import ContinualDataset
from utils.misc import read_pickle


def normalize(label, class_idx, upper=100.0):
    label_ranges = {
        1: (21.6, 102.6),
        2: (12.3, 16.87),
        3: (8.0, 50.0),
        4: (8.0, 50.0),
        5: (46.2, 104.88),
        6: (49.8, 99.36)
    }
    label_range = label_ranges[class_idx]

    norm_label = ((label - label_range[0]) / (label_range[1] - label_range[0])) * float(upper)
    return norm_label


class Seven_Dataset(torch.utils.data.Dataset):
    """AQA-7 dataset"""

    def __init__(self, args, subset, transform, samples):
        self.subset = subset
        self.transforms = transform

        classes_name = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
        self.sport_class = classes_name[args['class_idx'] - 1]

        self.class_idx = args['class_idx']  # sport class index(from 1 begin)
        self.score_range = args['score_range']
        # file path
        self.data_root = args['data_root']
        self.data_path = os.path.join(self.data_root, '{}-out'.format(self.sport_class))
        self.split_path = os.path.join(self.data_root, 'Split_4', f'split_4_{self.subset}_list.mat')
        self.split = scipy.io.loadmat(self.split_path)[f'consolidated_{self.subset}_list']
        self.split = samples

        self.dataset = self.split.copy()

        # setting
        self.length = args['frame_length']

    def load_video(self, idx):
        video_path = os.path.join(self.data_path, '%03d' % idx)
        video = [Image.open(os.path.join(video_path, 'img_%05d.jpg' % (i + 1))) for i in range(self.length)]
        return self.transforms(video)

    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        assert int(sample_1[0]) == self.class_idx
        idx = int(sample_1[1])

        data = self.load_video(idx)
        score = normalize(sample_1[2], self.class_idx, self.score_range)
        score = np.array(score).reshape(-1,)

        if self.subset == 'test':
            return data, score
        else:
            return data, score, data

    def __len__(self):
        return len(self.dataset)


def dataset_split(args, num_total_tasks, debug=False, fewshot=False, num_samples_per_task=20):
    data_root = args['data_root']
    class_idx = args['class_idx']
    score_range = args['score_range']

    classes_name = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
    sport_class = classes_name[class_idx - 1]

    split_path_train = os.path.join(data_root, 'Split_4', f'split_4_train_list.mat')
    train_split = scipy.io.loadmat(split_path_train)[f'consolidated_train_list']
    train_samples = train_split[train_split[:, 0] == class_idx].tolist()

    split_path_test = os.path.join(data_root, 'Split_4', f'split_4_test_list.mat')
    test_split = scipy.io.loadmat(split_path_test)[f'consolidated_test_list']
    test_samples = test_split[test_split[:, 0] == class_idx].tolist()

    train_labels = []
    for sample in train_samples:
        score = normalize(sample[2], class_idx, score_range)
        train_labels.append(score)

    test_labels = []
    for sample in test_samples:
        score = normalize(sample[2], class_idx, score_range)
        train_labels.append(score)

    # labels statistics
    labels = sorted(train_labels + test_labels)
    label_boundaries = [np.percentile(labels, p) for p in np.linspace(0, 100, num_total_tasks + 1)]

    # task split
    def split_task(samples, subset='training'):
        if subset == 'testing': num_samples_per_task = 10
        else: num_samples_per_task = 20

        def boundary_split(num_task):
            # left and right boundary
            left = label_boundaries[num_task]
            right = label_boundaries[num_task + 1]
            # ensure the last task contains the final sample
            if num_task == len(label_boundaries) - 2: right += 1

            boudary_samples, boudary_lables = [], []
            for sample in samples:
                label = normalize(sample[2], class_idx, score_range)
                if label >= left and label < right:
                    boudary_samples.append(sample)
                    boudary_lables.append(label)

            boudary_samples_rest = boudary_samples.copy()
            # sort all the labels
            def key_func(sample):
                return normalize(sample[2], class_idx, score_range)
            # the remaining samples as the base session
            if fewshot and len(boudary_samples) > num_samples_per_task:
                boudary_samples = sorted(boudary_samples, key=key_func)
                intervel = len(boudary_samples) / num_samples_per_task
                selected_idx = [int(i * intervel) for i in range(num_samples_per_task)]
                boudary_samples = [boudary_samples[idx] for idx in selected_idx]

                for idx, boudary_sample in enumerate(boudary_samples):
                    for idx_rest, sample_rest in enumerate(boudary_samples_rest):
                        if sample_rest == boudary_sample:
                            boudary_samples_rest.pop(idx_rest)
                # boudary_samples_rest = set(boudary_samples_rest) - set(boudary_samples)

                return boudary_samples, list(boudary_samples_rest)

            return boudary_samples

        if fewshot == False:
            sample_splits = [[] for _ in range(num_total_tasks)]

            for num_task in range(num_total_tasks):
                boudary_samples = boundary_split(num_task)
                sample_splits[num_task] += boudary_samples
        else:
            if int(os.environ.get("LOCAL_RANK", "-1")) < 1:
                print('Few-shot setting.')
            sample_splits = [[] for _ in range(num_total_tasks + 1)]

            for num_task in range(1, num_total_tasks + 1):
                boudary_samples, boudary_samples_rest = boundary_split(num_task - 1)
                sample_splits[num_task] += boudary_samples
                sample_splits[0] += boudary_samples_rest

        for num_task in range(len(sample_splits)):
            if int(os.environ.get("LOCAL_RANK", "-1")) < 1:
                if fewshot:
                    if num_task == 0:
                        num_samples = len(sample_splits[num_task])
                        print(f'Task {num_task}: {num_samples} samples ({subset}).')
                    else:
                        left = label_boundaries[num_task - 1]
                        right = label_boundaries[num_task]
                        num_samples = len(sample_splits[num_task])
                        print(f'Task {num_task} ({left:5.2f}，{right:6.2f}): {num_samples} samples ({subset}).')
                else:
                    left = label_boundaries[num_task]
                    right = label_boundaries[num_task + 1]
                    num_samples = len(sample_splits[num_task])
                    print(f'Task {num_task} ({left:5.2f}，{right:6.2f}): {num_samples} samples ({subset}).')

        return sample_splits

    train_sample_splits = split_task(train_samples)
    test_sample_splits = split_task(test_samples, subset='testing')

    return train_sample_splits, test_sample_splits


class ClassSeven(ContinualDataset):
    NAME = 'class-seven'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 5

    def __init__(self, args):
        super(ClassSeven, self).__init__(args)

        self.N_TASKS = self.args.n_tasks + 1 if self.args.fewshot else self.args.n_tasks
        self.train_sample_splits, self.test_sample_splits = \
            dataset_split(self.args.dataset_args, self.args.n_tasks, fewshot=self.args.fewshot)

        self.train_trans = video_transforms.Compose([
            video_transforms.RandomHorizontalFlip(),
            video_transforms.Resize((455, 256)),
            video_transforms.RandomCrop(224),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.test_trans = video_transforms.Compose([
            video_transforms.Resize((455, 256)),
            video_transforms.CenterCrop(224),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_joint_loader(self, subset='train'):
        if subset == 'train':
            transform = self.train_trans
            sample_splits = self.train_sample_splits

        else:
            transform = self.test_trans
            sample_splits = self.test_sample_splits

        # merge samples
        samples = []
        for i in range(self.i):
            samples += sample_splits[i]

        joint_loader = self.get_data_loader(transform, samples, subset)
        if int(os.environ.get("LOCAL_RANK", "-1")) < 1:
            print(f'The {subset} joint loader has {len(samples)} samples.')

        return joint_loader

    def get_data_loader(self, transform, samples, subset='train'):
        dataset = Seven_Dataset(self.args.dataset_args, transform=transform,
                              subset=subset, samples=samples)

        if self.args.local_rank != -1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=True if subset == 'train' else False)
        else:
            sampler = None

        loader = DataLoader(
            dataset, batch_size=self.args.batch_size,
            num_workers=self.args.num_workers, sampler=sampler,
            shuffle=True if self.args.local_rank == -1 and subset == 'train' else False)

        return loader

    def get_data_loaders(self):
        # data loaders
        self.train_loader = self.get_data_loader(transform=self.train_trans,
                                                 samples=self.train_sample_splits[self.i],
                                                 subset='train')
        self.test_loader = self.get_data_loader(transform=self.test_trans,
                                                samples=self.test_sample_splits[self.i],
                                                subset='test')

        self.train_loaders.append(self.train_loader)
        self.test_loaders.append(self.test_loader)

        self.i += 1

        return self.train_loader, self.test_loader

    def get_backbone(self):
        return AQAMLP(self.args)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_loss():

        def loss(output, target):
            output = output.to(torch.float32)
            target = target.to(torch.float32)

            if target.shape != output.shape:
                output_ = output * target[:, 1:]
                target_ = target[:, :1]

            else:
                output_, target_ = output, target

            return F.mse_loss(output_, target_)

        return loss

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_batch_size():
        return 128

    @staticmethod
    def get_minibatch_size():
        return 128
