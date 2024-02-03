import matplotlib.pyplot as plt
import os
import numpy as np
import random
import time
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchsummary import summary
from torcheval.metrics.functional import peak_signal_noise_ratio

SEED = 1
BATCH_SIZE = 8
IMG_HEIGHT = 256
IMG_WIDTH = 256

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

