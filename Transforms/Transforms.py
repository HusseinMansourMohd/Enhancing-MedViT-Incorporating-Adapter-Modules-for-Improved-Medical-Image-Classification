from transformers import DeformableDetrConfig, DeformableDetrModel

# Create a configuration for the Deformable DETR model
config = DeformableDetrConfig(
    use_timm_backbone=True,
    backbone_config=None,
    num_channels=3,
    num_queries=300,
    d_model=256,
    encoder_layers=6,
    decoder_layers=6,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    #dropout=0.1,
   
)

# Create a model using the configuration
ms_deform_attn = DeformableDetrModel(config)
