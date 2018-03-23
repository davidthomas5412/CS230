import os
import torch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn import MSELoss, NLLLoss
from torch.utils.data import Dataset, DataLoader

from time import time

for i in range(180000):
    a = np.load('/labs/khatrilab/scottmk/david/exp4/data/fifth/train/input/{}.npy'.format(i))
    if a.max() > 2:
        print(i)