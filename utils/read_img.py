# -*- coding: utf-8 -*-
# @Time: 2024/2/28 21:31
import cv2
import os
import pickle
import glob
import numpy as np
import math

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['savefig.bbox'] = 'tight'

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False

def read_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        pickle_data = pickle.load(f)

    return pickle_data

def plot_text(ax, txt):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x = (x_max - x_min) * 0.75 + x_min
    y = (y_max - y_min) * 0.8 + y_min

    plt.text(x, y, f'{txt}', va='center',
             ha='center', color='k', weight="bold")

def plot_desc(ax, desc, pos_x, pos_y):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x = (x_max - x_min) * pos_x + x_min
    y = (y_max - y_min) * pos_y + y_min

    plt.text(x, y, f'{desc}', va='center', ha='center', color='blue')

def norm_distribution(mu, sigma, fig_file):
    # rho
    fig = plt.figure(figsize=(2.6, 2.6), facecolor='#8FBBD9')
    ax = fig.add_subplot(111)

    x1 = np.linspace(mu - 10 * sigma, mu + 10 * sigma, 100)
    y1 = np.zeros_like(x1)
    y2 = np.exp(-1 * ((x1 - mu) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)

    if 'mer' not in fig_file:
        plt.plot(x1, y2, '#00B0F0', label='$\mu = ' + f'{mu:.2f}, \sigma = {sigma:.4f}$\n'
                                                      '$\hat{s} = \mu + \epsilon \\times \sigma$'
                                                      f'$\\approx{mu + sigma * np.random.rand(1)[0]:.2f}$\n'
                                                      f'$s=73.85$')
        plt.fill_between(x1, y1, y2, color='#00B0F0', alpha=0.5)

        plot_text(ax, 'MAGR\n(Ours)')
        # plot_desc(ax, '$\hat{s} = \mu + \\epsilon \\times \sigma$', 0.4, 0.2)
        # plot_desc(ax, '$~~ = \mu + \epsilon \times \sigma$', 0.4, 0.15)
    else:
        plt.plot(x1, y2, 'C1', label='$\mu = ' + f'{mu:.2f}, \sigma = {sigma:.4f}$\n'
                                                      '$\hat{s} = \mu + \epsilon \\times \sigma$'
                                                      f'$\\approx{mu + sigma * np.random.rand(1)[0]:.2f}$\n'
                                                      f'$s=73.85$')
        plt.fill_between(x1, y1, y2, color='C1', alpha=0.5)

        plot_text(ax, 'Feature\nMER')

    plt.xlabel('Score')
    plt.ylabel('Probability Density')

    plt.legend()
    plt.grid(axis='both', linestyle='-.', zorder=-100, color='silver')

    plt.savefig(fig_file)

if __name__ == '__main__':
    data_root = '/home/zkl/Documents/Data/AQA/Sequence/MTL-AQA/new'
    label_path = '/home/zkl/Documents/Data/AQA/Sequence/MTL-AQA/info/final_annotations_dict_with_dive_number.pkl'
    test_label_pkl = '/home/zkl/Documents/Data/AQA/Sequence/MTL-AQA/info/test_split_0.pkl'

    label_dict = read_pickle(label_path)
    test_split = read_pickle(test_label_pkl)
    gt_scores = [label_dict.get(s).get('final_score') for s in test_split]
    for idx, s in enumerate(gt_scores):
        if s == 73.95: print(idx)
    sample_idx = 124
    sample = test_split[sample_idx]
    gt_score = label_dict.get(sample).get('final_score')

    # load video
    image_list = sorted(
        (glob.glob(os.path.join(data_root, f'{sample[0]:02d}', '*.jpg')))
    )
    end_frame = label_dict.get(sample).get('end_frame')
    video = [cv2.imread(image_list[end_frame - 103 + i]) for i in range(103)]
    for i in [1, 21, 41, 61, 81, 101]:
        img_name = f'../outputs/figs/video_score/{sample_idx}-{i:02d}.jpg'
        os.makedirs(os.path.dirname(img_name), exist_ok=True)
        w, h, _ = video[i-1].shape
        print(w, h)
        cv2.imwrite(img_name, video[i-1][:, ((h - w) // 2):(- (h - w) // 2), :])
        print(f'Save {img_name}.')

    mu = 75.23
    sigma = 0.0050
    fig_name = f'../outputs/figs/video_score/{sample_idx}-mu_sigma.pdf'
    norm_distribution(mu, sigma, fig_name)
    print(gt_score)

    mu = 91.72
    sigma = 0.0125
    fig_name = f'../outputs/figs/video_score/{sample_idx}-mu_sigma-mer.pdf'
    norm_distribution(mu, sigma, fig_name)
    print(gt_score)

