from apex import amp
amp_handle = amp.init(enabled=True)
# %matplotlib inline
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from ssdoil import *
from ssd import *
from utils import *
import pandas as pd
from pathlib import Path

from fastai.vision.transform import get_transforms
from fastai.vision.data import ObjectItemList, imagenet_stats#, bb_pad_collate
from fastai import *
from fastai.vision import *

import sys

PATH = Path(r'./Bosch_trafficlight_data/')
TEST_JPEGS = 'train/TESTJPEGS'
IMG_PATH = PATH/TEST_JPEGS


classes = ["Green", "Red", "Yellow", "GreenLeft", "RedLeft", "RedRight", "RedStraight", "RedStraightLeft", "GreenRight", "GreenStraightRight", "GreenStraightLeft", "GreenStraight", "off"]



# sys.setrecursionlimit(15200)

# p = pickle.load(open(PATH/'train/AnnotatedJPEG/export.pkl', 'rb'))