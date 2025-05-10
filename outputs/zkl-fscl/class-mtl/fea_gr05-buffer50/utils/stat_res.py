# -*- coding: utf-8 -*-
# @Time : 23/02/2024 16:25

import json
import os
import re
import shutil
import numpy as np

def read_res(log_file_dir):
    with open(log_file_dir) as f:
        content = f.read()

    pos = content.find('Results:')
    if pos == -1: return None
    res = content[pos + 8:]
    res = res.replace("'", '"')
    res = res.replace("None", '-100')
    res = res.replace("nan", '-100')
    res = json.loads(res)

    for k, v in res.items():
        if v == -100: res[k] = np.inf

    pos = content.find('Rho (overall):', pos - 100)
    if pos == -1: return None
    pos_end = content.find('%', pos)
    rho = float(content[pos + 14:pos_end - 1])
    res['rho'] = rho

    return res

def get_results(log_file_dir):
    results = []
    log_files = []

    for root, dirs, files in os.walk(log_file_dir):
        for name in files:
            file = os.path.join(root, name)
            if file.endswith('.log'):
                log_files.append(file)

    for file in sorted(log_files):
        res = read_res(file)

        if res and '05' in file:
            print(os.path.basename(file)[6:-4], end='  & ')
            print(f"{file.split('/')[-3]:24s}", end='  & ')
            print(f"({res['rho'] / 100:6.4f}, {res['forgetting'] / 100:6.4f}, "
                  f"{res['fwt'] / 100:6.4f}) \\\\")
            results.append({'log_file': file,
                            'accs': res['accs'],
                            'mean_acc': np.mean(res['accs']),
                            'forgetting': res['forgetting'],
                            'fwt': res['fwt'],
                            'rho': res['rho']})
    return results

if __name__ == '__main__':
    user = os.environ['USER']
    print(user)
    if user == 'zkl':
        log_file_dir = f'/home/zkl/Documents/Codes/CL/aqa/outputs/zkl-fscl/class-mtl-noise'
        log_file_dir = f'/home/zkl/Documents/Codes/CL/aqa/outputs/zkl-fscl/class-mtl-dd'

    elif user == 'ubuntu':
        log_file_dir = f'/mnt/Codes/CL_AQA/outputs/ubuntu-fscl/class-mtl-noise'
        log_file_dir = f'/mnt/Codes/CL_AQA/outputs/ubuntu-fscl/class-mtl'

    else:
        raise ValueError(f'Unknown user: {user}')

    results = get_results(log_file_dir)