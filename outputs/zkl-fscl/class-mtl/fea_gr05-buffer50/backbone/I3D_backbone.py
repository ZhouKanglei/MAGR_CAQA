# -*- coding: utf-8 -*-
# @Time: 2023/6/23 21:55
import os

import torch

from backbone.I3D import I3D
from utils.misc import fix_bn

class I3D_backbone(torch.nn.Module):
    def __init__(self, I3D_ckpt_path, I3D_class=400):
        super(I3D_backbone, self).__init__()

        self.backbone = I3D(I3D_class)
        self.load_pretrain(I3D_ckpt_path)

    def load_pretrain(self, I3D_ckpt_path):

        self.backbone.load_state_dict(torch.load(I3D_ckpt_path), strict=False)
        if int(os.environ.get("LOCAL_RANK", "-1")) < 1:
            print('Using I3D backbone:')
            print(f'\tLoad pretrained model from {I3D_ckpt_path}.')

    def get_feature_dim(self):
        return self.backbone.get_logits_dim()

    def forward(self, video):
        batch_size, C, frames, H, W = video.shape

        # spatial-temporal feature
        if frames >= 160:
            start_idx = [i for i in range(0, frames, 16)]
        else:
            start_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 87]

        clips = [video[:, :, i:i + 16] for i in start_idx]

        clip_feats = torch.empty(batch_size, 1024, 10).to(video.device)

        for i in range(len(start_idx)):
            clip_feats[:, :, i] = self.backbone(clips[i]).reshape(batch_size, 1024)

        video_feats = clip_feats.mean(-1)
        # video_feats = video_feats / (torch.norm(video_feats, dim=-1, keepdim=True))

        return video_feats
