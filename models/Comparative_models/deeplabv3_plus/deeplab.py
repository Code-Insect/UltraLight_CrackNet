"""https://github.com/jfzhang95/pytorch-deeplab-xception/tree/master"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .aspp import build_aspp
from .decoder import build_decoder
from . import xception
from . import resnet


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet50':
        return resnet.ResNet50(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    else:
        raise NotImplementedError


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet50', output_stride=16, num_classes=1):
        super(DeepLab, self).__init__()

        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=False)

        return x


if __name__ == '__main__':
    model = DeepLab().cuda()
    x = torch.rand((4, 3, 544, 384)).cuda()
    y = model(x)
    print(y.size())


