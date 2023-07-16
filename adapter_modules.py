import logging
from functools import partial

import torch
import torch.nn as nn

from ops.modules import MSDeformAttn
from time.models.layers import DropPath
import torch.utils.checkpoint as cp


_logger = logging.getLogger(__name__)

def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_,W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5 , H_,dtype=torch.float32, device=device),
            torch.linspace(0.5, H_ - 0.5, W_, dtype=torch.float32, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x,ref_y), -1)
        reference_points_list = torch.cat(reference_points_list, 1)
        reference_points = torch.cat(reference_points_list,1)
        reference_points = reference_points[:,:,None] 
        return reference_points
    
