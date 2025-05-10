# -*- coding: utf-8 -*-
# @Time: 2024/3/2 14:02

import glob
import os
import pickle
import random
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvideotransforms import video_transforms, volume_transforms
from backbone.AQAMLP import AQAMLP
from datasets.utils.continual_dataset import ContinualDataset

class FineDiving_Dataset(torch.utils.data.Dataset):
    """MTL-AQA dataset"""
    def __init__(self, args, subset, transform, samples):
        # dataset
        self.subset = subset
        self.transform = transform
        self.dataset = samples
        # file path
        self.label_path = args['label_path']
        self.label_dict = self.read_pickle(self.label_path)

        self.data_root = args['data_root']

        # setting
        self.temporal_shift = args['temporal_shift']
        self.length = args['frame_length']

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)

        return pickle_data


    def load_video(self, video_file_name, phase):
        image_list = sorted(
            (glob.glob(os.path.join(self.data_root, video_file_name[0], str(video_file_name[1]), '*.jpg')))
        )

        start_frame = int(image_list[0].split("/")[-1][:-4])
        end_frame = int(image_list[-1].split("/")[-1][:-4])
        # if phase == 'train':
        #     temporal_aug_shift = random.randint(self.temporal_shift[0], self.temporal_shift[1])
        #     if end_frame + temporal_aug_shift > self.length or end_frame + temporal_aug_shift < len(image_list):
        #         end_frame = end_frame + temporal_aug_shift

        frame_list = np.linspace(start_frame, end_frame, self.length).astype(np.int64)
        image_frame_idx = [frame_list[i] - start_frame for i in range(self.length)]

        video = [Image.open(image_list[image_frame_idx[i]]) for i in range(self.length)]
        return self.transform(video)


    def __getitem__(self, index):

        sample = self.dataset[index]
        data = {}
        data['video'] = self.load_video(sample, self.subset)
        data['number'] = self.label_dict.get(sample)[0]
        data['final_score'] = self.label_dict.get(sample)[1]
        data['difficulty'] = self.label_dict.get(sample)[2]
        data['completeness'] = data['final_score'] / data['difficulty']

        if self.subset == 'test':
            return data['video'], np.array([data['final_score'], 1])
        else:
            return data['video'], np.array([data['final_score'], 1]), data['video']

    def __len__(self):
        return len(self.dataset)


def read_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        pickle_data = pickle.load(f)

    return pickle_data


def dataset_split(args, num_total_tasks, debug=False, fewshot=False, num_samples_per_task=20):

    split_path_test = args['test_split']
    split_path_train = args['train_split']
    label_path = args['label_path']

    label_dict = read_pickle(label_path)
    train_samples = read_pickle(split_path_train)
    test_samples = read_pickle(split_path_test)

    train_labels = []
    for sample in train_samples:
        train_labels.append(label_dict.get(sample)[1])

    test_labels = []
    for sample in test_samples:
        test_labels.append(label_dict.get(sample)[1])

    # labels statistics
    labels = sorted(train_labels + test_labels)
    label_boundaries = [np.percentile(labels, p) for p in np.linspace(0, 100, num_total_tasks + 1)]

    # task split
    def split_task(samples, subset='training'):

        num_samples_per_task_ = 20 if subset == 'testing' else num_samples_per_task
        # num_samples_per_task_ = num_samples_per_task

        def boundary_split(num_task):
            left = label_boundaries[num_task]
            right = label_boundaries[num_task + 1]
            if num_task == len(label_boundaries) - 2: right += 1

            boudary_samples, boudary_lables = [], []
            for sample in samples:
                label = label_dict.get(sample)[1]
                if label >= left and label < right:
                    boudary_samples.append(sample)
                    boudary_lables.append(label)

            boudary_samples_rest = boudary_samples.copy()

            def key_func(sample):
                return label_dict.get(sample)[1]

            if fewshot and len(boudary_samples) > num_samples_per_task_:
                boudary_samples = sorted(boudary_samples, key=key_func)
                intervel = math.floor(len(boudary_samples) / num_samples_per_task_)
                selected_idx = [i * intervel for i in range(num_samples_per_task_)]
                boudary_samples = [boudary_samples[idx] for idx in selected_idx]

                boudary_samples_rest = set(boudary_samples_rest) - set(boudary_samples)

                return boudary_samples, list(boudary_samples_rest)

            return boudary_samples, []

        if fewshot == False:
            sample_splits = [[] for _ in range(num_total_tasks)]

            for num_task in range(num_total_tasks):
                boudary_samples, _ = boundary_split(num_task)
                sample_splits[num_task] += boudary_samples
        else:
            if int(os.environ.get("LOCAL_RANK", "-1")) < 1:
                print('Few-shot setting.')
            sample_splits = [[] for _ in range(num_total_tasks + 1)]

            for num_task in range(1, num_total_tasks+1):
                boudary_samples, boudary_samples_rest = boundary_split(num_task - 1)
                sample_splits[num_task] += boudary_samples
                if len(boudary_samples_rest) == 0:
                    boudary_samples_rest = [boudary_samples[0]]
                    boudary_samples.pop(0)
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


class ClassFD(ContinualDataset):
    NAME = 'class-fd'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 5

    def __init__(self, args):
        super(ClassFD, self).__init__(args)

        self.N_TASKS = self.args.n_tasks + 1 if self.args.fewshot else self.args.n_tasks
        self.train_sample_splits, self.test_sample_splits = \
            dataset_split(self.args.dataset_args, self.args.n_tasks,
                          fewshot=self.args.fewshot,
                          num_samples_per_task=self.args.num_samples_per_task)

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

            all_samples = read_pickle(self.args.dataset_args['train_split'])
        else:
            transform = self.test_trans
            sample_splits = self.test_sample_splits

            all_samples = read_pickle(self.args.dataset_args['test_split'])

        all_labels = read_pickle(self.args.dataset_args['label_path'])

        # merge samples
        samples = []
        for i in range(self.i):
            samples += sample_splits[i]

        # print(subset,  '--- ', set(all_samples) - set(samples))
        # for sample in set(all_samples) - set(samples):
        #     print(all_labels.get(sample)[1])

        joint_loader = self.get_data_loader(transform, samples, subset)
        if int(os.environ.get("LOCAL_RANK", "-1")) < 1:
            print(f'The {subset} joint loader has {len(samples)} samples.')

        return joint_loader

    def get_data_loader(self, transform, samples, subset='train'):
        dataset = FineDiving_Dataset(self.args.dataset_args, transform=transform,
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