import torch 

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

import os, random, logging
from pathlib import Path
from helper_files.tools import Config as cfg
from helper_files.tools import start_logger
from randla.data_prep import data_loaders
from modules.net_modules70 import iNet, sth
from helper_files.data_utils import build_input1, unpack_input1, test_seg_alt
from tqdm import tqdm
import numpy as np
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from preformer.helper_files.metrics import *

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

ROOT_PATH = (Path(cfg.root)).resolve() # already at subsampled
# data_path = ROOT_PATH / 'subsampled'  

dataset_sampling = 'active_learning'
batch_size = 16
use_color = True

'''LOG'''
start_logger(cfg)
logging.info('point set size: {}'.format(cfg.num_points))

num_samples = torch.tensor(cfg.class_weights, dtype=torch.float, device='cuda:0')
ratio_samples = num_samples/num_samples.sum()
weights = 1 / (ratio_samples + 0.02)
print('Weights: ', weights)

seg_label_to_cat = {}
for i, cat in enumerate(cfg.class2label.keys()):
    seg_label_to_cat[i] = cat

tr_loader, val_loader = data_loaders(ROOT_PATH, cfg.sampling_type,
                                        batch_size=batch_size,
                                        use_color=use_color,
                                        num_workers=6,
                                        pin_memory=False)


print("done !")