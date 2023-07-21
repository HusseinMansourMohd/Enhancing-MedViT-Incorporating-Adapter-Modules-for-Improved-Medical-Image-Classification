

class MedVit_adapter(nn.Module): 
    def __init__(self, stem_chs, depths, path_dropout, attn_drop=0, drop=0, num_classes=1000,
                 strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=32, mix_block_ratio=0.75,
                 use_checkpoint=False, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0., interaction_indexes=None, with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, pretrained=None,
                 use_extra_extractor=True, with_cp=False, *args, **kwargs):

        super(MedVit_adapter, self).__init__(num_heads=num_heads , pretrained=pretrained)
        
        self._initialize_hyperparameters(path_dropout, use_checkpoint, pretrain_size, interaction_indexes, 
                                         num_heads, pretrained, use_extra_extractor, with_cp)
        
        self._initialize_embed_and_layers(embed_dim, conv_inplane, with_cp, deform_num_heads, 
                                          n_points, init_values, with_cffn, cffn_ratio, deform_ratio)

        self._initialize_stem_and_features(stem_chs, depths, strides, sr_ratios, head_dim, 
                                           mix_block_ratio, attn_drop, drop, path_dropout)
        
        self._initialize_final_layers(output_channel, num_classes)
        
        self._initialize_additional_layers(embed_dim)
        
        self._initialize_weights()

    def _initialize_hyperparameters(self, path_dropout, use_checkpoint, pretrain_size, 
                                    interaction_indexes, num_heads, pretrained, use_extra_extractor, with_cp):
        self.use_checkpoint = use_checkpoint
        self.embed_dim = self.embed_dim
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.num_heads = num_heads
        self.pretrained = pretrained
        self.use_extra_extractor = use_extra_extractor
        self.with_cp = with_cp

    def _initialize_embed_and_layers(self, embed_dim, conv_inplane, with_cp, deform_num_heads, 
                                     n_points, init_values, with_cffn, cffn_ratio, deform_ratio):
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=with_cp)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and self.use_extra_extractor),
                             with_cp=self.with_cp)
                             for i in range(len(self.interaction_indexes))
        ])

    def _initialize_stem_and_features(self, stem_chs, depths, strides, sr_ratios, head_dim, 
                                      mix_block_ratio, attn_drop, drop, path_dropout):
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

    def _create_features(self, depths, strides, sr_ratios, head_dim, 
                         mix_block_ratio, attn_drop, drop, path_dropout):
        input_channel = stem_chs[-1]
        features = []
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]  # stochastic depth decay rule
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = self.stage_out_channels[stage_id][block_id]
                block_type = self.stage_block_types[stage_id][block_id]
                layer = self._create_layer(block_type, input_channel, output_channel, 
                                           stride, dpr[idx + block_id], head_dim, 
                                           mix_block_ratio, attn_drop, drop, sr_ratios, stage_id)
                features.append(layer)
                input_channel = output_channel
        return nn.Sequential(*features)

    def _create_layer(self, block_type, input_channel, output_channel, 
                      stride, path_dropout, head_dim, mix_block_ratio, 
                      attn_drop, drop, sr_ratios, stage_id):
        if block_type is ECB:
            return ECB(input_channel, output_channel, stride=stride, 
                       path_dropout=path_dropout, drop=drop, head_dim=head_dim)
        elif block_type is LTB:
            return LTB(input_channel, output_channel, path_dropout=path_dropout, 
                       stride=stride, sr_ratio=sr_ratios[stage_id], head_dim=head_dim, 
                       mix_block_ratio=mix_block_ratio, attn_drop=attn_drop, drop=drop)

    def _initialize_final_layers(self, output_channel, num_classes):
        self.norm = nn.BatchNorm2d(output_channel, eps=NORM_EPS)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj_head = nn.Sequential(
            nn.Linear(output_channel, num_classes),
        )

    def _initialize_additional_layers(self, embed_dim):
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        def merge_bn(self):
        self.eval()
        for idx, module in self.named_modules():
            if isinstance(module, ECB) or isinstance(module, LTB):
                module.merge_bn()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        
    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1 , deform_inputs2 = deform_inputs(x)

        # SRM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        #Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:,1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Iteration
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x,c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                        deform_inputs1, deform_inputs2, H, W)
            outs.append(x.transpose(1,2).veiw(bs, dim, H, W).contiguous())

        #Split & Reshape
        c2 = c[:, 0:c2.size(1),:]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1), :]

        c2 = c2.transpose(1,2).view(bs, dim, H * 2, W *2).contiguous()
        c3 = c3.transpose(1,2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1,2).view(bs, dim, H//2, W//2).contiguous()

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        
        x = torch.cat([f1, f2, f3, f4], dim=1)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.proj_head(x) 

        return x
