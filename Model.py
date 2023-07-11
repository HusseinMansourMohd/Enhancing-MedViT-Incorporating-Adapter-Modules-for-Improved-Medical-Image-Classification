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


class PatchEmbed(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1
    ):
        super(PatchEmbed , self).__init__()
        norm_layer = partial(nn.BatchNorm, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.Avgpool2d((2,2), stride=2 , ceil_mode=True, count_include_pad=False)
            self.conv = nn.conv2d(in_channels, out_channels, kernel_size=1, stride=1, bais=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool))
    

class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """
    def __init__(self, out_channels , head_dim):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.group_conv3x3 == nn.conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                        padding=1, groups=out_channels // head_dim , bais=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.Relu(inplace=True)
        self.projection = nn.Conv2d(out_channels ,out_channels , kernel_size=1, bais=False)

    def foward(self , x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out
    
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x*self.sigmoid(x)
        

class h_swish(nn.Module):
    def __init__(self, implace=True):
        super(h_swish, self).__init__()
        self.relu = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
    

class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2 , b=1 , sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs(math.log(channel,2)+ b)/ gamma)
        k = t if t % 2 else t+1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.conv1d(1,1, kernel_size=k , padding=k // 2, bais=False)
        if sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = h_sigmoid()

    def foward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptivePool2d(1)
        self.fc = nn.Sequential(
            nn.linear(channel, channel // reduction)
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction , channel)
            h_sigmoid()
        )
    def forward(self, x):
        b, c _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b,c,1,1)
        return x*y
    

class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim , out_dim, stride, expand_ratio=4. , act='hs+se', reduction=4,
                 wp_dp_conv=False, dp_first = False):
         """
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
                    hs+se: h_swish and SE module
                    hs+eca: h_swish and ECA module
                    hs+ecah: h_swish and ECA module. Compared with eca, h_sigmoid is used.
        :param reduction: reduction rate in SE module.
        :param wo_dp_conv: without depth-wise convolution.
        :param dp_first: place depth-wise convolution as the first layer.
        """
         super(LocalityFeedForward, self).__init__()
         hidden_dim = int(in_dim * expand_ratio)
         kernel_size = 3

         layers = []
         # the first linear layer is replaced by 1x1 convolution.
         layers.extend([
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)])
         
         # the depth-wise conveloution between the two linear layers
         if not wo_dp_conv:
             dp =[
                 nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size// 2 , groups=hidden_dim, bais=False),
                 nn.BatchNorm2d(hidden_dim),
                 h_swish() if act.find('hs') >= 0 else nn.RelU6(inplace=True)
             ]
             if pd_first:
                 layers = dp + layers
             else:
                 layers.extend(dp)


         if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:
                 layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
            else:
                raise NotImplementedError('Activition type { is not implement}'.format(act))
            
            #the second linear layer is repaced by 1x1 convloution
            layers.extend(
                [
                    nn.Conv2d(hidden_dim, out_dim , 1, 1, 0, bais=False),
                    nn.BatchNorm2d(out_dim)
                ]
            )
            self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x =x + self.conv(x)
        return x
    

             
        




         
