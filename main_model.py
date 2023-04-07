import torch
from torch import nn as nn
from torch.nn import functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

from attention_layers import single_head_attention, multi_head_attention, Upsampling
















