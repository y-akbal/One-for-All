import os
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from memmap_arrays import ts_concatted
import numpy as np
import time
from main_model import Model
## import hydra now
import hydra
from omegaconf import DictConfig, OmegaConf

## This is important in the case that you compile the model!!!!
torch.set_float32_matmul_precision("high")

