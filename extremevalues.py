import os
import torch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn import MSELoss, NLLLoss
from torch.utils.data import Dataset, DataLoader

from time import time

def toscalar(var):
    return np.mean(var.data.cpu().numpy())

def train(config):
    np.random.seed(2)
    torch.manual_seed(2)
    if config.cuda:
        torch.cuda.manual_seed(0)

    model = config.model
    model.eval()
    model.cuda()
    model.load_state_dict(torch.load('experiments/checkpoint_paper5_best'))

    # for params in model.parameters():
    #     if (params > 10).any():
    #         print(params.max())

    xcounter = 0
    for i, sample in enumerate(config.dataloader()):
        input_batch = Variable(sample['input'].cuda(async=True))
        label_batch = Variable(sample['label'].cuda(async=True))

        # forward pass
        output_batch = model(input_batch)

        if (output_batch > 20).any():
            print('issue')
            output = np.squeeze(output_batch.data.cpu().numpy())
            input = np.squeeze(input_batch.data.cpu().numpy())
            label = np.squeeze(label_batch.data.cpu().numpy())
            np.save('data/extreme5/{}_out.npy'.format(xcounter), output)
            np.save('data/extreme5/{}_lab.npy'.format(xcounter), label)
            np.save('data/extreme5/{}_inp.npy'.format(xcounter), input)
            xcounter += 1
        if i % 1000 == 0:
            print(i)


def isvariable(label):
    return np.sum(label[0,50:-50,50:-50] > 1) > 2

def isnonvariable(label):
    return np.isclose(np.sum(label),0)

class AugmentedData(Dataset):

    TRAIN_SIZE = 10000 #2 * 9000

    def __init__(self):
        self.input_folder = os.path.join('data', 'fifth', 'train', 'input')
        self.label_folder = os.path.join('data', 'fifth', 'train', 'label')

    def __len__(self):
        return AugmentedData.TRAIN_SIZE

    def __getitem__(self, idx):
        input_file = os.path.join(self.input_folder, '{}.npy'.format(idx))
        label_file = os.path.join(self.label_folder, '{}.npy'.format(idx))
        inp = np.load(input_file).astype('float32') * np.log(10 ** 5)
        lab = np.load(label_file).astype('float32') * np.log(10 ** 5)
        inp = inp.reshape(-1, inp.shape[0], inp.shape[1])
        lab = lab.reshape(-1, lab.shape[0], lab.shape[1])

        output = {'input': torch.from_numpy(inp), 
                  'label': torch.from_numpy(lab)
                  }
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

        return img

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

class SimpleConfig:
    def __init__(self):
        self.batch_size = 1
        self.epochs = 1
        self.cuda = True
        self.pin_memory = True
        self.model = Net()


    def dataloader(self):
        return DataLoader(AugmentedData(), 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=1,
                pin_memory=self.pin_memory)


if __name__ == '__main__':
    config = SimpleConfig()
    train(config)
