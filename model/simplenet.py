# Went with https://github.com/meetshah1995/pytorch-semseg/tree/master/ptsemseg/models
# because of batchnorm

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeeperBNNet(nn.Module):
    def __init__(self):
        super(DeeperBNNet, self).__init__()
        self.first = double_convbn(1, 64)
        self.down1 = downbn(64, 128)
        self.down2 = downbn(128, 256)
        self.down3 = downbn(256, 256)
        self.up1 = upbn(512, 128)
        self.up2 = upbn(256, 64)
        self.up3 = upbn(128, 64)
        self.last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.last(x)
        return x

class HalfNet(nn.Module):
    def __init__(self):
        super(HalfNet, self).__init__()
        self.first = half(1, 64)
        self.down1 = downhalf(64, 128)
        self.down2 = downhalf(128, 256)
        self.down3 = downhalf(256, 256)
        self.up1 = uphalf(512, 128)
        self.up2 = uphalf(256, 64)
        self.up3 = uphalf(128, 64)
        self.last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.last(x)
        return x

class half(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(half, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class downhalf(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downhalf, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            half(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x 


class uphalf(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(uphalf, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = half(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class SevenConvNet(nn.Module):
    def __init__(self):
        super(SevenConvNet, self).__init__()
        self.first = double_convseven(1, 64)
        self.down1 = downseven(64, 128)
        self.down2 = downseven(128, 256)
        self.down3 = downseven(256, 256)
        self.up1 = upseven(512, 128)
        self.up2 = upseven(256, 64)
        self.up3 = upseven(128, 64)
        self.last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.last(x)
        return x

class double_convseven(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_convseven, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 7, padding=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class downseven(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downseven, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_convseven(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x 


class upseven(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upseven, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = double_convseven(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x



class FiveConvNet(nn.Module):
    def __init__(self):
        super(FiveConvNet, self).__init__()
        self.first = double_convbig(1, 64)
        self.down1 = downbig(64, 128)
        self.down2 = downbig(128, 256)
        self.down3 = downbig(256, 256)
        self.up1 = upbig(512, 128)
        self.up2 = upbig(256, 64)
        self.up3 = upbig(128, 64)
        self.last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.last(x)
        return x

class double_convbig(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_convbig, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 5, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class downbig(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downbig, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_convbig(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x 


class upbig(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upbig, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = double_convbig(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class DeeperBNDropoutNet(nn.Module):
    def __init__(self):
        super(DeeperBNDropoutNet, self).__init__()
        self.first = double_convbn(1, 64)
        self.down1 = downbn(64, 128)
        self.down2 = downbn(128, 256)
        self.drop1 = nn.Dropout(p=0.3)
        self.down3 = downbn(256, 256)
        self.up1 = upbn(512, 128)
        self.drop2 = nn.Dropout(p=0.3)
        self.up2 = upbn(256, 64)
        self.up3 = upbn(128, 64)
        self.last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.drop1(x3)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.drop2(x)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.last(x)
        return x

class DeeperNet(nn.Module):
    def __init__(self):
        super(DeeperNet, self).__init__()
        self.first = double_conv(1, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 64)
        self.last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.last(x)
        return x

class SimpleBatchnormNet(nn.Module):
    def __init__(self):
        super(SimpleBatchnormNet, self).__init__()
        self.first = double_convbn(1, 64)
        self.down1 = downbn(64, 128)
        self.down2 = downbn(128, 128)
        # self.down3 = down(256, 256)
        # self.up1 = up(512, 256)
        self.up1 = upbn(256, 64)
        self.up2 = upbn(128, 32)
        self.last = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.last(x)
        return x


class double_convbn(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_convbn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class downbn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downbn, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_convbn(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x 


class upbn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upbn, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = double_convbn(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.first = double_conv(1, 64)
#         self.down1 = down(64, 128)
#         self.down2 = down(128, 128)
#         # self.down3 = down(256, 256)
#         # self.up1 = up(512, 256)
#         self.up1 = up(256, 64)
#         self.up2 = up(128, 32)
#         self.last = nn.Conv2d(32, 1, 1)

#     def forward(self, x):
#         x1 = self.first(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x = self.up1(x3, x2)
#         x = self.up2(x, x1)
#         x = self.last(x)
#         return x

#  (nn.Module):
#     '''(conv => BN => ReLU) * 2'''
#     def __init__(self, in_ch, out_ch):
#         super(double_conv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class down(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(down, self).__init__()
#         self.mpconv = nn.Sequential(
#             nn.MaxPool2d(2),
#             double_conv(in_ch, out_ch)
#         )

#     def forward(self, x):
#         x = self.mpconv(x)
#         return x 


# class up(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(up, self).__init__()
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.conv = double_conv(in_ch, out_ch)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         diffX = x1.size()[2] - x2.size()[2]
#         diffY = x1.size()[3] - x2.size()[3]
#         x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
#                         diffY // 2, int(diffY / 2)))
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x