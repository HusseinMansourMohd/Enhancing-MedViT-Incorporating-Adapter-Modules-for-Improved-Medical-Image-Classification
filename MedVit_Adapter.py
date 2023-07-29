from functools import partial
from torch import nn
import numpy as np
from adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs
import torch
from MedVit import ConvBNReLU,ECB,LTB,PatchEmbed
from timm.models.registry import register_model
import torch.nn.functional as F

NORM_EPS= 1e-5

class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.avgpool(x)
        H, W = x.size(2), x.size(3)
        x = self.norm(self.conv(x))
        return x, H, W


class MedVit_adapter(nn.Module): 
    def __init__(self, embed_dim, stem_chs, depths, path_dropout, attn_drop=0, drop=0, num_classes=1000,
                 strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=32, mix_block_ratio=0.75,
                 use_checkpoint=False, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0., interaction_indexes=None, with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, pretrained=None, output_channel=512,
                 use_extra_extractor=True, with_cp=False, *args, **kwargs):

        super(MedVit_adapter, self).__init__()

        self.embed_dim = embed_dim
        self.stem_chs = stem_chs
        input_channel = stem_chs[-1]
        
        self.patch_embed = PatchEmbed(in_channels=input_channel, out_channels = 512)

        self._initialize_hyperparameters(path_dropout, use_checkpoint, pretrain_size, interaction_indexes, 
                                         num_heads, pretrained, use_extra_extractor, with_cp)

        self._initialize_embed_and_layers(conv_inplane, deform_num_heads, 
                                          n_points, init_values, with_cffn, cffn_ratio, deform_ratio=1.0)

        self._initialize_stem_and_features(stem_chs, depths, strides, sr_ratios, head_dim, 
                                           mix_block_ratio, attn_drop, drop, path_dropout)
        self.norm = nn.BatchNorm2d(output_channel, eps=NORM_EPS)
        self._initialize_final_layers(output_channel, num_classes)

    def _initialize_hyperparameters(self, path_dropout, use_checkpoint, pretrain_size, 
                                    interaction_indexes, num_heads, pretrained, use_extra_extractor, with_cp):
        self.use_checkpoint = use_checkpoint
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.num_heads = num_heads
        self.pretrained = pretrained
        self.use_extra_extractor = use_extra_extractor
        self.with_cp = with_cp

    def _initialize_embed_and_layers(self, conv_inplane , deform_num_heads, 
                                     n_points, init_values, with_cffn, cffn_ratio, deform_ratio ):
        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=self.embed_dim)
        
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=self.embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len([]) - 1
                                               else False) and self.use_extra_extractor),
                             with_cp=self.with_cp)
                             for i in range(len([]))
        ])

    def _create_features(self, depths, strides, sr_ratios, head_dim, mix_block_ratio, attn_drop, drop, path_dropout):
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]  # stochastic depth decay rule
        input_channel = self.stem_chs[-1]
        stem_chs = self.stem_chs
        features = []
        idx = 0
        self.stage_out_channels = [[96] * (depths[0]),
                                   [192] * (depths[1] - 1) + [256],
                                   [384, 384, 384, 384, 512] * (depths[2] // 5),
                                   [768] * (depths[3] - 1) + [1024]]

        #  Hybrid Strategy
        self.stage_block_types = [[ECB] * depths[0],
                                  [ECB] * (depths[1] - 1) + [LTB],
                                  [ECB, ECB, ECB, ECB, LTB] * (depths[2] // 5),
                                  [ECB] * (depths[3] - 1) + [LTB]]

        self.stem = nn.Sequential(
            ConvBNReLU(3, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )

        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is ECB:
                    layer = ECB(input_channel, output_channel, stride=stride, path_dropout=dpr[idx + block_id],
                                drop=drop, head_dim=head_dim)
                    features.append(layer)
                elif block_type is LTB:
                    layer = LTB(input_channel, output_channel, path_dropout=dpr[idx + block_id], stride=stride,
                                sr_ratio=sr_ratios[stage_id], head_dim=head_dim, mix_block_ratio=mix_block_ratio,
                                attn_drop=attn_drop, drop=drop)
                    features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        return nn.Sequential(*features)

    def _initialize_stem_and_features(self, stem_chs, depths, strides, sr_ratios, head_dim, 
                                      mix_block_ratio, attn_drop, drop, path_dropout):
        stem_chs = self.stem_chs
        self.stem = self._create_stem(stem_chs)
        self.features = self._create_features(depths, strides, sr_ratios, head_dim, 
                                              mix_block_ratio, attn_drop, drop, path_dropout)

    def _create_stem(self, stem_chs):
        return nn.Sequential(
            ConvBNReLU(3, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )

    def _initialize_final_layers(self, output_channel, num_classes):
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj_head = nn.Sequential(
            nn.Linear(output_channel, num_classes),
        )
    
    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4
    
    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def forward(self, x):
        deform_inputs1 , deform_inputs2 = deform_inputs(x)

        c1, c2, c3, c4 = self.spm(x)
        x = self.stem(x)
        # SRM forward
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Embedding and Position embedding
        x, H, W = self.patch_embed(x)
        dim = x.shape
        bs = x.shape
        pos_embed = np.identity(4)
        if(pos_embed):
            pos_embed = self._get_pos_embed(self.pos_embed[:, 1:] , H, W)
        else:
            pos_embed = np.Ide
            pos_embed = self._get_pos_embed(pos_embed, H, W)
        x = self.pos_drop(x + pos_embed)

        # Interactions
        outs = []
        for i, interaction in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = interaction(x, c, self.blocks[indexes[0]:indexes[-1] + 1], deform_inputs1, deform_inputs2, H, W)
            outs.append(x.transpose(1,2).view(bs, dim, H, W).contiguous())

        # Reshaping
        c2 = c[:, 0:c2.size(1),:]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1), :]
        c2 = c2.transpose(1,2).view(bs, dim, H * 2, W *2).contiguous()
        c3 = c3.transpose(1,2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1,2).view(bs, dim, H//2, W//2).contiguous()

        # Feature interpolation and addition
        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm and output
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        x = torch.cat([f1, f2, f3, f4], dim=1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.proj_head(x)

        return x

@register_model
def MedViT_adapter_small(pretrained=False, pretrained_cfg=None, **kwargs):
    model = MedVit_adapter(embed_dim=64, stem_chs=[64, 32, 64], depths=[3, 4, 10, 3], path_dropout=0.1, **kwargs)
    return model

@register_model
def MedViT_adapter_base(pretrained=False, pretrained_cfg=None, **kwargs):
    model = MedVit_adapter(embed_dim=64, stem_chs=[64, 32, 64], depths=[3, 4, 20, 3], path_dropout=0.2, **kwargs)
    return model

@register_model
def MedViT_adapter_large(pretrained=False, pretrained_cfg=None, **kwargs):
    model = MedVit_adapter(embed_dim=64, stem_chs=[64, 32, 64], depths=[3, 4, 30, 3], path_dropout=0.2, **kwargs)
    return model


