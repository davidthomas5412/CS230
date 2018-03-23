import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(FirstPaperNet, self).__init__()
        self.first = DoubleConvLayer(1, 64)
        self.down1 = DownLevel(64, 128)
        self.dropout1 = nn.Dropout(p=0.1)
        self.down2 = DownLevel(128, 256)
        self.down3 = DownLevel(256, 256)
        self.dropout2 = nn.Dropout(p=0.1)
        self.up1 = UpLevel(512, 128)
        self.up2 = UpLevel(256, 64)
        self.dropout3 = nn.Dropout(p=0.1)
        self.up3 = UpLevel(128, 32)
        self.last = nn.Conv2d(32, 1, 1)
        self.binary = BinaryClassifier()

    def forward(self, x):
        xd1 = self.first(x)
        xd2 = self.down1(xd1)
        xd2 = self.dropout1(xd2)
        xd3 = self.down2(xd2)
        xd4 = self.down3(xd3)
        xd4 = self.dropout2(xd4)
        xu1 = self.up1(xd4, xd3)
        xu2 = self.up2(xu1, xd2)
        xu2 = self.dropout3(xu2)
        xu3 = self.up3(xu2, xd1)
        
        img = self.last(xu3)
        pred = self.binary(x, img)

        return img, pred

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.binary = nn.Sequential(
            nn.Conv2d(2, 16, 4, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 4, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 4, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 4, stride=1),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, orig, intermediate):
        x = torch.cat([orig, intermediate], dim=1)
        return self.binary(x)

class DoubleConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConvLayer, self).__init__()
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

class DownLevel(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownLevel, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvLayer(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x 


class UpLevel(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpLevel, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = DoubleConvLayer(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x