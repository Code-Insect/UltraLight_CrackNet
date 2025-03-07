"""https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py"""
from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out


class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """
    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out


class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out


class R2U_Net(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)

        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)

        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.RRCNN1(x)

        e2 = self.Maxpool(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool1(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool3(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

      # out = self.active(out)

        return out


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out


class R2AttU_Net(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """
    def __init__(self, in_ch=3, out_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = RRCNN_block(in_ch, filters[0], t=t)
        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)
        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)
        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)
        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.RRCNN1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        e4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out

#For nested 3 channels are required

class conv_block_nested(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output
    
#Nested Unet

class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(NestedUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output

#Dictioary Unet
#if required for getting the filters and model parameters for each step 

class ConvolutionBlock(nn.Module):
    """Convolution block"""

    def __init__(self, in_filters, out_filters, kernel_size=3, batchnorm=True, last_active=F.relu):
        super(ConvolutionBlock, self).__init__()

        self.bn = batchnorm
        self.last_active = last_active
        self.c1 = nn.Conv2d(in_filters, out_filters, kernel_size, padding=1)
        self.b1 = nn.BatchNorm2d(out_filters)
        self.c2 = nn.Conv2d(out_filters, out_filters, kernel_size, padding=1)
        self.b2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.c1(x)
        if self.bn:
            x = self.b1(x)
        x = F.relu(x)
        x = self.c2(x)
        if self.bn:
            x = self.b2(x)
        x = self.last_active(x)
        return x


class ContractiveBlock(nn.Module):
    """Deconvuling Block"""

    def __init__(self, in_filters, out_filters, conv_kern=3, pool_kern=2, dropout=0.5, batchnorm=True):
        super(ContractiveBlock, self).__init__()
        self.c1 = ConvolutionBlock(in_filters=in_filters, out_filters=out_filters, kernel_size=conv_kern,
                                   batchnorm=batchnorm)
        self.p1 = nn.MaxPool2d(kernel_size=pool_kern, ceil_mode=True)
        self.d1 = nn.Dropout2d(dropout)

    def forward(self, x):
        c = self.c1(x)
        return c, self.d1(self.p1(c))


class ExpansiveBlock(nn.Module):
    """Upconvole Block"""

    def __init__(self, in_filters1, in_filters2, out_filters, tr_kern=3, conv_kern=3, stride=2, dropout=0.5):
        super(ExpansiveBlock, self).__init__()
        self.t1 = nn.ConvTranspose2d(in_filters1, out_filters, tr_kern, stride=2, padding=1, output_padding=1)
        self.d1 = nn.Dropout(dropout)
        self.c1 = ConvolutionBlock(out_filters + in_filters2, out_filters, conv_kern)

    def forward(self, x, contractive_x):
        x_ups = self.t1(x)
        x_concat = torch.cat([x_ups, contractive_x], 1)
        x_fin = self.c1(self.d1(x_concat))
        return x_fin


class Unet_dict(nn.Module):
    """Unet which operates with filters dictionary values"""

    def __init__(self, n_labels, n_filters=32, p_dropout=0.5, batchnorm=True):
        super(Unet_dict, self).__init__()
        filters_dict = {}
        filt_pair = [3, n_filters]

        for i in range(4):
            self.add_module('contractive_' + str(i), ContractiveBlock(filt_pair[0], filt_pair[1], batchnorm=batchnorm))
            filters_dict['contractive_' + str(i)] = (filt_pair[0], filt_pair[1])
            filt_pair[0] = filt_pair[1]
            filt_pair[1] = filt_pair[1] * 2

        self.bottleneck = ConvolutionBlock(filt_pair[0], filt_pair[1], batchnorm=batchnorm)
        filters_dict['bottleneck'] = (filt_pair[0], filt_pair[1])

        for i in reversed(range(4)):
            self.add_module('expansive_' + str(i),
                            ExpansiveBlock(filt_pair[1], filters_dict['contractive_' + str(i)][1], filt_pair[0]))
            filters_dict['expansive_' + str(i)] = (filt_pair[1], filt_pair[0])
            filt_pair[1] = filt_pair[0]
            filt_pair[0] = filt_pair[0] // 2

        self.output = nn.Conv2d(filt_pair[1], n_labels, kernel_size=1)
        filters_dict['output'] = (filt_pair[1], n_labels)
        self.filters_dict = filters_dict

    # final_forward
    def forward(self, x):
        c00, c0 = self.contractive_0(x)
        c11, c1 = self.contractive_1(c0)
        c22, c2 = self.contractive_2(c1)
        c33, c3 = self.contractive_3(c2)
        bottle = self.bottleneck(c3)
        u3 = F.relu(self.expansive_3(bottle, c33))
        u2 = F.relu(self.expansive_2(u3, c22))
        u1 = F.relu(self.expansive_1(u2, c11))
        u0 = F.relu(self.expansive_0(u1, c00))
        return F.softmax(self.output(u0), dim=1)


"""
SegNet https://github.com/yassouali/pytorch-segmentation/blob/master/models/segnet.py
"""
from models.Comparative_models.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy


class SegNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, pretrained=True, freeze_bn=False, **_):
        super(SegNet, self).__init__()
        vgg_bn = models.vgg16_bn(weights='VGG16_BN_Weights.IMAGENET1K_V1')
        encoder = list(vgg_bn.features.children())

        # Adjust the input size
        if in_channels != 3:
            encoder[0] = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1)

        # Encoder, VGG without any maxpooling
        self.stage1_encoder = nn.Sequential(*encoder[:6])
        self.stage2_encoder = nn.Sequential(*encoder[7:13])
        self.stage3_encoder = nn.Sequential(*encoder[14:23])
        self.stage4_encoder = nn.Sequential(*encoder[24:33])
        self.stage5_encoder = nn.Sequential(*encoder[34:-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder, same as the encoder but reversed, maxpool will not be used
        # 2023.11.2  Serious error!!! Shallow copy of the list, where elements within the list point to the same object, and many modules (especially the BN layer) may be duplicated
        # decoder = encoder

        # List deep copy
        decoder = copy.deepcopy(encoder)

        decoder = [i for i in list(reversed(decoder))
                   if not isinstance(i, nn.MaxPool2d)]
        # Replace the last conv layer
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # When reversing, we also reversed conv->batchN->relu, correct it
        decoder = [item for i in range(0, len(decoder), 3)
                   for item in decoder[i:i+3][::-1]]
        # Replace some conv layers & batchN after them
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                if module.in_channels != module.out_channels:
                    decoder[i+1] = nn.BatchNorm2d(module.in_channels)
                    decoder[i] = nn.Conv2d(
                        module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1)

        self.stage1_decoder = nn.Sequential(*decoder[0:9])
        self.stage2_decoder = nn.Sequential(*decoder[9:18])
        self.stage3_decoder = nn.Sequential(*decoder[18:27])
        self.stage4_decoder = nn.Sequential(*decoder[27:33])
        self.stage5_decoder = nn.Sequential(*decoder[33:],
                                            nn.Conv2d(
                                                64, num_classes, kernel_size=3, stride=1, padding=1)
                                            )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self._initialize_weights(self.stage1_decoder, self.stage2_decoder, self.stage3_decoder,
                                 self.stage4_decoder, self.stage5_decoder)
        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        # Encoder
        x = self.stage1_encoder(x)
        x1_size = x.size()
        x, indices1 = self.pool(x)

        x = self.stage2_encoder(x)
        x2_size = x.size()
        x, indices2 = self.pool(x)

        x = self.stage3_encoder(x)
        x3_size = x.size()
        x, indices3 = self.pool(x)

        x = self.stage4_encoder(x)
        x4_size = x.size()
        x, indices4 = self.pool(x)

        x = self.stage5_encoder(x)
        x5_size = x.size()
        x, indices5 = self.pool(x)

        # Decoder
        x = self.unpool(x, indices=indices5, output_size=x5_size)
        x = self.stage1_decoder(x)

        x = self.unpool(x, indices=indices4, output_size=x4_size)
        x = self.stage2_decoder(x)

        x = self.unpool(x, indices=indices3, output_size=x3_size)
        x = self.stage3_decoder(x)

        x = self.unpool(x, indices=indices2, output_size=x2_size)
        x = self.stage4_decoder(x)

        x = self.unpool(x, indices=indices1, output_size=x1_size)
        x = self.stage5_decoder(x)

        return x

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


"""
DeepCrack https://github.com/qinnzou/DeepCrack/blob/master/codes/model/deepcrack.py
"""
from torch import nn
import torch
import torch.nn.functional as F


def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv3X3(in_, out)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class Down(nn.Module):

    def __init__(self, nn):
        super(Down,self).__init__()
        self.nn = nn
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self,inputs):
        down = self.nn(inputs)
        unpooled_shape = down.size()
        outputs, indices = self.maxpool_with_argmax(down)
        return outputs, down, indices, unpooled_shape

class Up(nn.Module):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        self.unpool=torch.nn.MaxUnpool2d(2,2)

    def forward(self,inputs,indices,output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        outputs = self.nn(outputs)
        return outputs

class Fuse(nn.Module):

    def __init__(self, nn, scale):
        super().__init__()
        self.nn = nn
        self.scale = scale
        self.conv = Conv3X3(64,1)

    def forward(self,down_inp,up_inp):
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = F.interpolate(outputs, scale_factor=self.scale, mode='bilinear')
        outputs = self.nn(outputs)

        return self.conv(outputs)


class DeepCrack(nn.Module):

    def __init__(self):
        super(DeepCrack, self).__init__()

        self.down1 = Down(torch.nn.Sequential(
            ConvRelu(3,64),
            ConvRelu(64,64),
        ))

        self.down2 = Down(torch.nn.Sequential(
            ConvRelu(64,128),
            ConvRelu(128,128),
        ))

        self.down3 = Down(torch.nn.Sequential(
            ConvRelu(128,256),
            ConvRelu(256,256),
            ConvRelu(256,256),
        ))

        self.down4 = Down(torch.nn.Sequential(
            ConvRelu(256, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 512),
        ))

        self.down5 = Down(torch.nn.Sequential(
            ConvRelu(512, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 512),
        ))

        self.up1 = Up(torch.nn.Sequential(
            ConvRelu(64, 64),
            ConvRelu(64, 64),
        ))

        self.up2 = Up(torch.nn.Sequential(
            ConvRelu(128, 128),
            ConvRelu(128, 64),
        ))

        self.up3 = Up(torch.nn.Sequential(
            ConvRelu(256, 256),
            ConvRelu(256, 256),
            ConvRelu(256, 128),
        ))

        self.up4 = Up(torch.nn.Sequential(
            ConvRelu(512, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 256),
        ))

        self.up5 = Up(torch.nn.Sequential(
            ConvRelu(512, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 512),
        ))

        self.fuse5 = Fuse(ConvRelu(512 + 512, 64), scale=16)
        self.fuse4 = Fuse(ConvRelu(512 + 256, 64), scale=8)
        self.fuse3 = Fuse(ConvRelu(256 + 128, 64), scale=4)
        self.fuse2 = Fuse(ConvRelu(128 + 64, 64), scale=2)
        self.fuse1 = Fuse(ConvRelu(64 + 64, 64), scale=1)

        self.final = Conv3X3(5,1)

    def forward(self,inputs):

        # encoder part
        out, down1, indices_1, unpool_shape1 = self.down1(inputs)
        out, down2, indices_2, unpool_shape2 = self.down2(out)
        out, down3, indices_3, unpool_shape3 = self.down3(out)
        out, down4, indices_4, unpool_shape4 = self.down4(out)
        out, down5, indices_5, unpool_shape5 = self.down5(out)

        # decoder part
        up5 = self.up5(out, indices=indices_5, output_shape=unpool_shape5)
        up4 = self.up4(up5, indices=indices_4, output_shape=unpool_shape4)
        up3 = self.up3(up4, indices=indices_3, output_shape=unpool_shape3)
        up2 = self.up2(up3, indices=indices_2, output_shape=unpool_shape2)
        up1 = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)

        fuse5 = self.fuse5(down_inp=down5,up_inp=up5)
        fuse4 = self.fuse4(down_inp=down4, up_inp=up4)
        fuse3 = self.fuse3(down_inp=down3, up_inp=up3)
        fuse2 = self.fuse2(down_inp=down2, up_inp=up2)
        fuse1 = self.fuse1(down_inp=down1, up_inp=up1)

        output = self.final(torch.cat([fuse5,fuse4,fuse3,fuse2,fuse1],1))

        return output


"""
RCF https://github.com/yun-liu/RCF-PyTorch/blob/master/models.py
"""
import scipy.io as sio
import numpy as np


class RCF(nn.Module):
    def __init__(self, pretrained=None):
        super(RCF, self).__init__()
        self.conv1_1 = nn.Conv2d(  3,  64, 3, padding=1, dilation=1)
        self.conv1_2 = nn.Conv2d( 64,  64, 3, padding=1, dilation=1)
        self.conv2_1 = nn.Conv2d( 64, 128, 3, padding=1, dilation=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1, dilation=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1, dilation=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)
        self.act = nn.ReLU(inplace=True)

        self.conv1_1_down = nn.Conv2d( 64, 21, 1)
        self.conv1_2_down = nn.Conv2d( 64, 21, 1)
        self.conv2_1_down = nn.Conv2d(128, 21, 1)
        self.conv2_2_down = nn.Conv2d(128, 21, 1)
        self.conv3_1_down = nn.Conv2d(256, 21, 1)
        self.conv3_2_down = nn.Conv2d(256, 21, 1)
        self.conv3_3_down = nn.Conv2d(256, 21, 1)
        self.conv4_1_down = nn.Conv2d(512, 21, 1)
        self.conv4_2_down = nn.Conv2d(512, 21, 1)
        self.conv4_3_down = nn.Conv2d(512, 21, 1)
        self.conv5_1_down = nn.Conv2d(512, 21, 1)
        self.conv5_2_down = nn.Conv2d(512, 21, 1)
        self.conv5_3_down = nn.Conv2d(512, 21, 1)

        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        self.score_fuse = nn.Conv2d(5, 1, 1)

        self.weight_deconv2 = self._make_bilinear_weights( 4, 1).cuda()
        self.weight_deconv3 = self._make_bilinear_weights( 8, 1).cuda()
        self.weight_deconv4 = self._make_bilinear_weights(16, 1).cuda()
        self.weight_deconv5 = self._make_bilinear_weights(16, 1).cuda()

        # init weights
        self.apply(self._init_weights)
        if pretrained is not None:
            vgg16 = sio.loadmat(pretrained)
            torch_params = self.state_dict()

            for k in vgg16.keys():
                name_par = k.split('-')
                size = len(name_par)
                if size == 2:
                    name_space = name_par[0] + '.' + name_par[1]
                    data = np.squeeze(vgg16[k])
                    torch_params[name_space] = torch.from_numpy(data)
            self.load_state_dict(torch_params)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
            if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
                nn.init.constant_(m.weight, 0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Based on HED implementation @ https://github.com/xwjabc/hed
    def _make_bilinear_weights(self, size, num_channels):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        w.requires_grad = False
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    w[i, j] = filt
        return w

    # Based on BDCN implementation @ https://github.com/pkuCactus/BDCN
    def _crop(self, data, img_h, img_w, crop_h, crop_w):
        _, _, h, w = data.size()
        assert(img_h <= h and img_w <= w)
        data = data[:, :, crop_h:crop_h + img_h, crop_w:crop_w + img_w]
        return data

    def forward(self, x):
        img_h, img_w = x.shape[2], x.shape[3]
        conv1_1 = self.act(self.conv1_1(x))
        conv1_2 = self.act(self.conv1_2(conv1_1))
        pool1   = self.pool1(conv1_2)
        conv2_1 = self.act(self.conv2_1(pool1))
        conv2_2 = self.act(self.conv2_2(conv2_1))
        pool2   = self.pool2(conv2_2)
        conv3_1 = self.act(self.conv3_1(pool2))
        conv3_2 = self.act(self.conv3_2(conv3_1))
        conv3_3 = self.act(self.conv3_3(conv3_2))
        pool3   = self.pool3(conv3_3)
        conv4_1 = self.act(self.conv4_1(pool3))
        conv4_2 = self.act(self.conv4_2(conv4_1))
        conv4_3 = self.act(self.conv4_3(conv4_2))
        pool4   = self.pool4(conv4_3)
        conv5_1 = self.act(self.conv5_1(pool4))
        conv5_2 = self.act(self.conv5_2(conv5_1))
        conv5_3 = self.act(self.conv5_3(conv5_2))

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)

        out1 = self.score_dsn1(conv1_1_down + conv1_2_down)
        out2 = self.score_dsn2(conv2_1_down + conv2_2_down)
        out3 = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        out4 = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        out5 = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)

        out2 = F.conv_transpose2d(out2, self.weight_deconv2, stride=2)
        out3 = F.conv_transpose2d(out3, self.weight_deconv3, stride=4)
        out4 = F.conv_transpose2d(out4, self.weight_deconv4, stride=8)
        out5 = F.conv_transpose2d(out5, self.weight_deconv5, stride=8)

        out2 = self._crop(out2, img_h, img_w, 1, 1)
        out3 = self._crop(out3, img_h, img_w, 2, 2)
        out4 = self._crop(out4, img_h, img_w, 4, 4)
        out5 = self._crop(out5, img_h, img_w, 0, 0)

        fuse = torch.cat((out1, out2, out3, out4, out5), dim=1)
        fuse = self.score_fuse(fuse)
        # results = [out1, out2, out3, out4, out5, fuse]
        # results = [torch.sigmoid(r) for r in results]
        return fuse


"""
HED https://github.com/xwjabc/hed/blob/master/networks.py
"""


class HED(nn.Module):
    """ HED network. """

    def __init__(self, device=torch.device('cuda')):
        super(HED, self).__init__()
        # Layers.
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=35)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.relu = nn.ReLU()
        # Note: ceil_mode – when True, will use ceil instead of floor to compute the output shape.
        #       The reason to use ceil mode here is that later we need to upsample the feature maps and crop the results
        #       in order to have the same shape as the original image. If ceil mode is not used, the up-sampled feature
        #       maps will possibly be smaller than the original images.
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.score_dsn1 = nn.Conv2d(64, 1, 1)  # Out channels: 1.
        self.score_dsn2 = nn.Conv2d(128, 1, 1)
        self.score_dsn3 = nn.Conv2d(256, 1, 1)
        self.score_dsn4 = nn.Conv2d(512, 1, 1)
        self.score_dsn5 = nn.Conv2d(512, 1, 1)
        self.score_final = nn.Conv2d(5, 1, 1)

        # Fixed bilinear weights.
        self.weight_deconv2 = make_bilinear_weights(4, 1).to(device)
        self.weight_deconv3 = make_bilinear_weights(8, 1).to(device)
        self.weight_deconv4 = make_bilinear_weights(16, 1).to(device)
        self.weight_deconv5 = make_bilinear_weights(32, 1).to(device)

        # Prepare for aligned crop.
        self.crop1_margin, self.crop2_margin, self.crop3_margin, self.crop4_margin, self.crop5_margin = \
            self.prepare_aligned_crop()

    # noinspection PyMethodMayBeStatic
    def prepare_aligned_crop(self):
        """ Prepare for aligned crop. """

        # Re-implement the logic in deploy.prototxt and
        #   /hed/src/caffe/layers/crop_layer.cpp of official repo.
        # Other reference materials:
        #   hed/include/caffe/layer.hpp
        #   hed/include/caffe/vision_layers.hpp
        #   hed/include/caffe/util/coords.hpp
        #   https://groups.google.com/forum/#!topic/caffe-users/YSRYy7Nd9J8

        def map_inv(m):
            """ Mapping inverse. """
            a, b = m
            return 1 / a, -b / a

        def map_compose(m1, m2):
            """ Mapping compose. """
            a1, b1 = m1
            a2, b2 = m2
            return a1 * a2, a1 * b2 + b1

        def deconv_map(kernel_h, stride_h, pad_h):
            """ Deconvolution coordinates mapping. """
            return stride_h, (kernel_h - 1) / 2 - pad_h

        def conv_map(kernel_h, stride_h, pad_h):
            """ Convolution coordinates mapping. """
            return map_inv(deconv_map(kernel_h, stride_h, pad_h))

        def pool_map(kernel_h, stride_h, pad_h):
            """ Pooling coordinates mapping. """
            return conv_map(kernel_h, stride_h, pad_h)

        x_map = (1, 0)
        conv1_1_map = map_compose(conv_map(3, 1, 35), x_map)
        conv1_2_map = map_compose(conv_map(3, 1, 1), conv1_1_map)
        pool1_map = map_compose(pool_map(2, 2, 0), conv1_2_map)

        conv2_1_map = map_compose(conv_map(3, 1, 1), pool1_map)
        conv2_2_map = map_compose(conv_map(3, 1, 1), conv2_1_map)
        pool2_map = map_compose(pool_map(2, 2, 0), conv2_2_map)

        conv3_1_map = map_compose(conv_map(3, 1, 1), pool2_map)
        conv3_2_map = map_compose(conv_map(3, 1, 1), conv3_1_map)
        conv3_3_map = map_compose(conv_map(3, 1, 1), conv3_2_map)
        pool3_map = map_compose(pool_map(2, 2, 0), conv3_3_map)

        conv4_1_map = map_compose(conv_map(3, 1, 1), pool3_map)
        conv4_2_map = map_compose(conv_map(3, 1, 1), conv4_1_map)
        conv4_3_map = map_compose(conv_map(3, 1, 1), conv4_2_map)
        pool4_map = map_compose(pool_map(2, 2, 0), conv4_3_map)

        conv5_1_map = map_compose(conv_map(3, 1, 1), pool4_map)
        conv5_2_map = map_compose(conv_map(3, 1, 1), conv5_1_map)
        conv5_3_map = map_compose(conv_map(3, 1, 1), conv5_2_map)

        score_dsn1_map = conv1_2_map
        score_dsn2_map = conv2_2_map
        score_dsn3_map = conv3_3_map
        score_dsn4_map = conv4_3_map
        score_dsn5_map = conv5_3_map

        upsample2_map = map_compose(deconv_map(4, 2, 0), score_dsn2_map)
        upsample3_map = map_compose(deconv_map(8, 4, 0), score_dsn3_map)
        upsample4_map = map_compose(deconv_map(16, 8, 0), score_dsn4_map)
        upsample5_map = map_compose(deconv_map(32, 16, 0), score_dsn5_map)

        crop1_margin = int(score_dsn1_map[1])
        crop2_margin = int(upsample2_map[1])
        crop3_margin = int(upsample3_map[1])
        crop4_margin = int(upsample4_map[1])
        crop5_margin = int(upsample5_map[1])

        return crop1_margin, crop2_margin, crop3_margin, crop4_margin, crop5_margin

    def forward(self, x):
        # VGG-16 network.
        image_h, image_w = x.shape[2], x.shape[3]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))  # Side output 1.
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))  # Side output 2.
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))  # Side output 3.
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))  # Side output 4.
        pool4 = self.maxpool(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))  # Side output 5.

        score_dsn1 = self.score_dsn1(conv1_2)
        score_dsn2 = self.score_dsn2(conv2_2)
        score_dsn3 = self.score_dsn3(conv3_3)
        score_dsn4 = self.score_dsn4(conv4_3)
        score_dsn5 = self.score_dsn5(conv5_3)

        upsample2 = torch.nn.functional.conv_transpose2d(score_dsn2, self.weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(score_dsn3, self.weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(score_dsn4, self.weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(score_dsn5, self.weight_deconv5, stride=16)

        # Aligned cropping.
        crop1 = score_dsn1[:, :, self.crop1_margin:self.crop1_margin + image_h,
                self.crop1_margin:self.crop1_margin + image_w]
        crop2 = upsample2[:, :, self.crop2_margin:self.crop2_margin + image_h,
                self.crop2_margin:self.crop2_margin + image_w]
        crop3 = upsample3[:, :, self.crop3_margin:self.crop3_margin + image_h,
                self.crop3_margin:self.crop3_margin + image_w]
        crop4 = upsample4[:, :, self.crop4_margin:self.crop4_margin + image_h,
                self.crop4_margin:self.crop4_margin + image_w]
        crop5 = upsample5[:, :, self.crop5_margin:self.crop5_margin + image_h,
                self.crop5_margin:self.crop5_margin + image_w]

        # Concatenate according to channels.
        fuse_cat = torch.cat((crop1, crop2, crop3, crop4, crop5), dim=1)
        fuse = self.score_final(fuse_cat)  # Shape: [batch_size, 1, image_h, image_w].
        # results = [crop1, crop2, crop3, crop4, crop5, fuse]
        # results = [torch.sigmoid(r) for r in results]
        return fuse


def make_bilinear_weights(size, num_channels):
    """ Generate bi-linear interpolation weights as up-sampling filters (following FCN paper). """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False  # Set not trainable.
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


if __name__ == '__main__':
    model = SegNet(num_classes=1).cuda()
    x = torch.rand((4, 3, 544, 384)).cuda()
    y = model(x)
    print(y.size())


