# -*- coding: utf-8 -*-
# @Time: 2023/6/20 22:31
import os


def get_all_configs():
    return [config.split('.')[0] for config in os.listdir('configs')
            if not config.find('__') > -1 and 'yaml' in config]
