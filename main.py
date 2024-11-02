# -*- coding: utf-8 -*-
# @Time: 2023/6/20 22:42

import warnings
warnings.filterwarnings("ignore")

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.parser import Parser
from utils.processor_aqa import ProcessorAQA as Processor

if __name__ == '__main__':
    args = Parser().args

    processor = Processor(args)

    processor.start()
