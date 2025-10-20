================================================================================
TORCHVISION MODEL DISCOVERY
================================================================================

Total models in torchvision: 121
Skip patterns: rcnn, retinanet, fcos, ssd, deeplabv3, fcn, lraspp, raft, r3d, r2plus1d, mc3, s3d, mvit, swin3d, quantized


================================================================================
SUMMARY
================================================================================

✓ FX-traceable:  80 models
✗ Failed:        0 models
⊘ Skipped:       41 models (detection/segmentation/video/quantized)

================================================================================
FX-TRACEABLE MODELS BY FAMILY
================================================================================

ALEXNET (1):
  alexnet

CONVNEXT (4):
  convnext_base
  convnext_large
  convnext_small
  convnext_tiny

DENSENET121 (1):
  densenet121

DENSENET161 (1):
  densenet161

DENSENET169 (1):
  densenet169

DENSENET201 (1):
  densenet201

EFFICIENTNET (11):
  efficientnet_b0
  efficientnet_b1
  efficientnet_b2
  efficientnet_b3
  efficientnet_b4
  efficientnet_b5
  efficientnet_b6
  efficientnet_b7
  efficientnet_v2_l
  efficientnet_v2_m
  efficientnet_v2_s

GOOGLENET (1):
  googlenet

INCEPTION (1):
  inception_v3

MAXVIT (1):
  maxvit_t

MNASNET0 (2):
  mnasnet0_5
  mnasnet0_75

MNASNET1 (2):
  mnasnet1_0
  mnasnet1_3

MOBILENET (3):
  mobilenet_v2
  mobilenet_v3_large
  mobilenet_v3_small

REGNET (15):
  regnet_x_16gf
  regnet_x_1_6gf
  regnet_x_32gf
  regnet_x_3_2gf
  regnet_x_400mf
  regnet_x_800mf
  regnet_x_8gf
  regnet_y_128gf
  regnet_y_16gf
  regnet_y_1_6gf
  regnet_y_32gf
  regnet_y_3_2gf
  regnet_y_400mf
  regnet_y_800mf
  regnet_y_8gf

RESNET101 (1):
  resnet101

RESNET152 (1):
  resnet152

RESNET18 (1):
  resnet18

RESNET34 (1):
  resnet34

RESNET50 (1):
  resnet50

RESNEXT101 (2):
  resnext101_32x8d
  resnext101_64x4d

RESNEXT50 (1):
  resnext50_32x4d

SHUFFLENET (4):
  shufflenet_v2_x0_5
  shufflenet_v2_x1_0
  shufflenet_v2_x1_5
  shufflenet_v2_x2_0

SQUEEZENET1 (2):
  squeezenet1_0
  squeezenet1_1

SWIN (6):
  swin_b
  swin_s
  swin_t
  swin_v2_b
  swin_v2_s
  swin_v2_t

VGG11 (2):
  vgg11
  vgg11_bn

VGG13 (2):
  vgg13
  vgg13_bn

VGG16 (2):
  vgg16
  vgg16_bn

VGG19 (2):
  vgg19
  vgg19_bn

VIT (5):
  vit_b_16
  vit_b_32
  vit_h_14
  vit_l_16
  vit_l_32

WIDE (2):
  wide_resnet101_2
  wide_resnet50_2

================================================================================
GENERATED REGISTRY CODE
================================================================================

Copy this to replace MODEL_REGISTRY in profile_graph.py:

MODEL_REGISTRY = {
    # Alexnet family
    'alexnet': models.alexnet,
    # Convnext family
    'convnext_base': models.convnext_base,
    'convnext_large': models.convnext_large,
    'convnext_small': models.convnext_small,
    'convnext_tiny': models.convnext_tiny,
    # Densenet121 family
    'densenet121': models.densenet121,
    # Densenet161 family
    'densenet161': models.densenet161,
    # Densenet169 family
    'densenet169': models.densenet169,
    # Densenet201 family
    'densenet201': models.densenet201,
    # Efficientnet family
    'efficientnet_b0': models.efficientnet_b0,
    'efficientnet_b1': models.efficientnet_b1,
    'efficientnet_b2': models.efficientnet_b2,
    'efficientnet_b3': models.efficientnet_b3,
    'efficientnet_b4': models.efficientnet_b4,
    'efficientnet_b5': models.efficientnet_b5,
    'efficientnet_b6': models.efficientnet_b6,
    'efficientnet_b7': models.efficientnet_b7,
    'efficientnet_v2_l': models.efficientnet_v2_l,
    'efficientnet_v2_m': models.efficientnet_v2_m,
    'efficientnet_v2_s': models.efficientnet_v2_s,
    # Googlenet family
    'googlenet': models.googlenet,
    # Inception family
    'inception_v3': models.inception_v3,
    # Maxvit family
    'maxvit_t': models.maxvit_t,
    # Mnasnet0 family
    'mnasnet0_5': models.mnasnet0_5,
    'mnasnet0_75': models.mnasnet0_75,
    # Mnasnet1 family
    'mnasnet1_0': models.mnasnet1_0,
    'mnasnet1_3': models.mnasnet1_3,
    # Mobilenet family
    'mobilenet_v2': models.mobilenet_v2,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    'mobilenet_v3_small': models.mobilenet_v3_small,
    # Regnet family
    'regnet_x_16gf': models.regnet_x_16gf,
    'regnet_x_1_6gf': models.regnet_x_1_6gf,
    'regnet_x_32gf': models.regnet_x_32gf,
    'regnet_x_3_2gf': models.regnet_x_3_2gf,
    'regnet_x_400mf': models.regnet_x_400mf,
    'regnet_x_800mf': models.regnet_x_800mf,
    'regnet_x_8gf': models.regnet_x_8gf,
    'regnet_y_128gf': models.regnet_y_128gf,
    'regnet_y_16gf': models.regnet_y_16gf,
    'regnet_y_1_6gf': models.regnet_y_1_6gf,
    'regnet_y_32gf': models.regnet_y_32gf,
    'regnet_y_3_2gf': models.regnet_y_3_2gf,
    'regnet_y_400mf': models.regnet_y_400mf,
    'regnet_y_800mf': models.regnet_y_800mf,
    'regnet_y_8gf': models.regnet_y_8gf,
    # Resnet101 family
    'resnet101': models.resnet101,
    # Resnet152 family
    'resnet152': models.resnet152,
    # Resnet18 family
    'resnet18': models.resnet18,
    # Resnet34 family
    'resnet34': models.resnet34,
    # Resnet50 family
    'resnet50': models.resnet50,
    # Resnext101 family
    'resnext101_32x8d': models.resnext101_32x8d,
    'resnext101_64x4d': models.resnext101_64x4d,
    # Resnext50 family
    'resnext50_32x4d': models.resnext50_32x4d,
    # Shufflenet family
    'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
    'shufflenet_v2_x1_5': models.shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0': models.shufflenet_v2_x2_0,
    # Squeezenet1 family
    'squeezenet1_0': models.squeezenet1_0,
    'squeezenet1_1': models.squeezenet1_1,
    # Swin family
    'swin_b': models.swin_b,
    'swin_s': models.swin_s,
    'swin_t': models.swin_t,
    'swin_v2_b': models.swin_v2_b,
    'swin_v2_s': models.swin_v2_s,
    'swin_v2_t': models.swin_v2_t,
    # Vgg11 family
    'vgg11': models.vgg11,
    'vgg11_bn': models.vgg11_bn,
    # Vgg13 family
    'vgg13': models.vgg13,
    'vgg13_bn': models.vgg13_bn,
    # Vgg16 family
    'vgg16': models.vgg16,
    'vgg16_bn': models.vgg16_bn,
    # Vgg19 family
    'vgg19': models.vgg19,
    'vgg19_bn': models.vgg19_bn,
    # Vit family
    'vit_b_16': models.vit_b_16,
    'vit_b_32': models.vit_b_32,
    'vit_h_14': models.vit_h_14,
    'vit_l_16': models.vit_l_16,
    'vit_l_32': models.vit_l_32,
    # Wide family
    'wide_resnet101_2': models.wide_resnet101_2,
    'wide_resnet50_2': models.wide_resnet50_2,
}
