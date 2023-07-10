from functools import partial
import math
import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, tunc_normal_
from timm.models.registry import register_model
from torch import nn
from utils import merge_pre_bn

NORM_EPS = 1e-5

class convBNReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            group=1
    ):
        super(convBNReLU, self).__init__()
        self.conv = nn.conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride ,padding=1, groups=groups , bais=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    

    def _make_divisbele(v, divior, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value ,int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

        
