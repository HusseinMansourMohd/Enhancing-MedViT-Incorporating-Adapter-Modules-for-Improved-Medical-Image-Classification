# import logging
# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # from mmdet.models.builder import BACKBONES
# # from ops.modules import MSDeformAttn
# # from timm.models.layers import DropPath, trunc_normal_
# # from torch.nn.init import normal_

# # from .base.vit import TIMMVisionTransformer

# from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs
# from MedVit import LTB, ECB, ConvBNReLU

# class MedViTAdapter(nn.Module):
#     def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4, deform_num_heads=6,
#                  init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
#                  deform_ratio=1.0, add_vit_feature=True, use_extra_extractor=True, *args, **kwargs):

#         super().__init__(num_heads=num_heads, *args, **kwargs)

#         # Initialization of MedViT-specific parameters goes here...
#         self.use_checkpoint = use_checkpoint

#         self.stage_out_channels = [[96] * (depths[0]),
#                                    [192] * (depths[1] - 1) + [256],
#                                    [384, 384, 384, 384, 512] * (depths[2] // 5),
#                                    [768] * (depths[3] - 1) + [1024]]

#         # Next Hybrid Strategy
#         self.stage_block_types = [[ECB] * depths[0],
#                                   [ECB] * (depths[1] - 1) + [LTB],
#                                   [ECB, ECB, ECB, ECB, LTB] * (depths[2] // 5),
#                                   [ECB] * (depths[3] - 1) + [LTB]]

#         self.stem = nn.Sequential(
#             ConvBNReLU(3, stem_chs[0], kernel_size=3, stride=2),
#             ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
#             ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
#             ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
#         )
#         input_channel = stem_chs[-1]
#         features = []
#         idx = 0
#         dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]  # stochastic depth decay rule
#         for stage_id in range(len(depths)):
#             numrepeat = depths[stage_id]
#             output_channels = self.stage_out_channels[stage_id]
#             block_types = self.stage_block_types[stage_id]
#             for block_id in range(numrepeat):
#                 if strides[stage_id] == 2 and block_id == 0:
#                     stride = 2
#                 else:
#                     stride = 1
#                 output_channel = output_channels[block_id]
#                 block_type = block_types[block_id]
#                 if block_type is ECB:
#                     layer = ECB(input_channel, output_channel, stride=stride, path_dropout=dpr[idx + block_id],
#                                 drop=drop, head_dim=head_dim)
#                     features.append(layer)
#                 elif block_type is LTB:
#                     layer = LTB(input_channel, output_channel, path_dropout=dpr[idx + block_id], stride=stride,
#                                 sr_ratio=sr_ratios[stage_id], head_dim=head_dim, mix_block_ratio=mix_block_ratio,
#                                 attn_drop=attn_drop, drop=drop)
#                     features.append(layer)
#                 input_channel = output_channel
#             idx += numrepeat
#         self.features = nn.Sequential(*features)

#         self.norm = nn.BatchNorm2d(output_channel, eps=NORM_EPS)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.proj_head = nn.Sequential(
#             nn.Linear(output_channel, num_classes),
#         )

#         self.stage_out_idx = [sum(depths[:idx + 1]) - 1 for idx in range(len(depths))]
#         print('initialize_weights...')
#         self._initialize_weights()

#     def merge_bn(self):
#         self.eval()
#         for idx, module in self.named_modules():
#             if isinstance(module, ECB) or isinstance(module, LTB):
#                 module.merge_bn()

#     def _initialize_weights(self):
#         for n, m in self.named_modules():
#             if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 trunc_normal_(m.weight, std=.02)
#                 if hasattr(m, 'bias') and m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Conv2d):
#                 trunc_normal_(m.weight, std=.02)
#                 if hasattr(m, 'bias') and m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#     # Initialization of MedViT-specific parameters goes here...







#         # InteractionBlocks
#         self.interactions = nn.Sequential(*[
#             InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
#                              init_values=init_values, drop_path=self.drop_path_rate,
#                              norm_layer=self.norm_layer, with_cffn=with_cffn,
#                              cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
#                              extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor))
#             for i in range(len(interaction_indexes))
#         ])

#     # Forward method for MedViT
#     def forward(self, x):
#         # Preprocess the input as in MedViT...
        
#         # Additional pre-processing required for injectors and extractors
#         deform_inputs1, deform_inputs2 = deform_inputs(x)
#         c1, c2, c3, c4 = self.spm(x)
#         c2, c3, c4 = self._add_level_embed(c2, c3, c4)
#         c = torch.cat([c2, c3, c4], dim=1)

#         # Interaction
#         for i, layer in enumerate(self.interactions):
#             indexes = self.interaction_indexes[i]
#             x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
#                          deform_inputs1, deform_inputs2, H, W)

#         # Continue with the rest of the forward method from MedViT...

#     def forward(self, x):
#         x = self.stem(x)
#         for idx, layer in enumerate(self.features):
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(layer, x)
#             else:
#                 x = layer(x)
#         x = self.norm(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.proj_head(x)
#         return x
