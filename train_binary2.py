import os
import torch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

def train(config):
    np.random.seed(0)
    torch.manual_seed(0)
    if config.cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.set_device(1)
    model = config.model
    model.train()
    model.cuda()
    optimizer = config.optimizer

    ce_loss = nn.CrossEntropyLoss()
    best_loss = 10

    for epoch in range(config.epochs):

        for i, sample in enumerate(config.dataloader(mode='train')):
            input_batch = Variable(sample['input'].cuda(async=True))
            label_batch = Variable(sample['label'].cuda(async=True))
            label_batch = label_batch.view(-1)

            # forward pass
            output_batch = model(input_batch)

            loss = ce_loss(output_batch, label_batch)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % config.steps_per_train_loss == 0:
                acc = accuracy(output_batch, label_batch)
                message = '[{}, {}] ce: {}, acc: {}'.format(epoch, i, toscalar(loss), toscalar(acc))
                print(message)

            if i % config.steps_per_dev_loss == 0:
                model.eval()
                acc_arr = []
                dloss_arr = []
                for j, dev_sample in enumerate(config.dataloader(mode='dev')):
                    input_batch = Variable(dev_sample['input'].cuda(async=True))
                    label_batch = Variable(dev_sample['label'].cuda(async=True))
                    label_batch = label_batch.view(-1)

                    output_batch = model(input_batch)
                    acc_arr.append(toscalar(accuracy(output_batch, label_batch)))
                    dloss_arr.append(toscalar(ce_loss(output_batch, label_batch)))
                avg_dloss = np.mean(dloss_arr)
                avg_acc = np.mean(acc_arr)
                message = '[{}, {}] avg_ce: {}, avg_acc: {}'.format(epoch, i, avg_dloss, avg_acc)
                print(message)

                if avg_dloss < best_loss:
                    print('best')
                    best_loss = avg_dloss
                    state = model.state_dict()
                    torch.save(state, config.best_checkpoint_path)

                model.train()

            if i % config.steps_per_save == 0:
                state = model.state_dict()
                torch.save(state, config.checkpoint_path)

        # write checkpoint after each epoch
        state = model.state_dict()
        torch.save(state, config.checkpoint_path)


def accuracy(output, label):
    """Computes the accuracy for multiple binary predictions"""
    pred = output.max(1)[1]
    acc = pred.eq(label).sum().float() / label.size(0)
    return acc

def toscalar(var):
    return np.mean(var.data.cpu().numpy())

class TrainData(Dataset):
    nonvar_count = 6810
    var_count = 3176

    def __init__(self):
        self.var_folder = os.path.join('data', 'binary', 'train', 'var')
        self.nonvar_folder = os.path.join('data', 'binary', 'train', 'nonvar')

    def __len__(self):
        return TrainData.nonvar_count + TrainData.var_count # var + nonvar

    def __getitem__(self, idx):
        if idx >= TrainData.var_count:
            idx = idx - TrainData.var_count
            input_file = os.path.join(self.nonvar_folder, '{}.npy'.format(idx))
            inp = np.load(input_file).astype('float32')
            inp = inp.reshape(-1, inp.shape[0], inp.shape[1])
            lab = np.array([0], dtype='int64')

        else:
            input_file = os.path.join(self.var_folder, '{}.npy'.format(idx))
            inp = np.load(input_file).astype('float32')
            inp = inp.reshape(-1, inp.shape[0], inp.shape[1])
            lab = np.array([1], dtype='int64')

        output = {'input': torch.from_numpy(inp), 
                  'label': torch.from_numpy(lab)
                  }

        return output

class DevData(Dataset):
    nonvar_count = 69
    var_count = 31

    def __init__(self):
        self.var_folder = os.path.join('data', 'binary', 'dev', 'var')
        self.nonvar_folder = os.path.join('data', 'binary', 'dev', 'nonvar')

    def __len__(self):
        return DevData.nonvar_count + DevData.var_count # var + nonvar

    def __getitem__(self, idx):
        if idx >= DevData.var_count:
            idx = idx - DevData.var_count
            input_file = os.path.join(self.nonvar_folder, '{}.npy'.format(idx))
            inp = np.load(input_file).astype('float32')
            inp = inp.reshape(-1, inp.shape[0], inp.shape[1])
            lab = np.array([0], dtype='int64')

        else:
            input_file = os.path.join(self.var_folder, '{}.npy'.format(idx))
            inp = np.load(input_file).astype('float32')
            inp = inp.reshape(-1, inp.shape[0], inp.shape[1])
            lab = np.array([1], dtype='int64')

        output = {'input': torch.from_numpy(inp), 
                  'label': torch.from_numpy(lab)
                  }

        return output

class TestData(Dataset):
    nonvar_count = 71
    var_count = 28

    def __init__(self):
        self.var_folder = os.path.join('data', 'binary', 'dev', 'var')
        self.nonvar_folder = os.path.join('data', 'binary', 'dev', 'nonvar')

    def __len__(self):
        return DevData.nonvar_count + DevData.var_count # var + nonvar

    def __getitem__(self, idx):
        if idx >= DevData.var_count:
            idx = idx - DevData.var_count
            input_file = os.path.join(self.nonvar_folder, '{}.npy'.format(idx))
            inp = np.load(input_file).astype('float32')
            inp = inp.reshape(-1, inp.shape[0], inp.shape[1])
            lab = np.array([0], dtype='int64')

        else:
            input_file = os.path.join(self.var_folder, '{}.npy'.format(idx))
            inp = np.load(input_file).astype('float32')
            inp = inp.reshape(-1, inp.shape[0], inp.shape[1])
            lab = np.array([1], dtype='int64')

        output = {'input': torch.from_numpy(inp), 
                  'label': torch.from_numpy(lab)
                  }

        return output

class BinaryNet(nn.Module):
    def __init__(self):
        super(BinaryNet, self).__init__()
        self.avg = nn.AvgPool2d(16, stride=8, padding=0)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=4, padding=2)
        self.max = nn.MaxPool2d(8, stride=8, padding=0)
        self.conv2 = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x):
        x = self.avg(x)
        x = self.conv1(x)
        x = self.max(x)
        x = self.conv2(x)
        return x.view(-1,2)

class BinaryNet2(nn.Module):
    def __init__(self):
        super(BinaryNet2, self).__init__()
        self.mp1 = nn.MaxPool2d(4, stride=4, padding=0)
        self.cv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.r1 = nn.ReLU(inplace=True)
        self.mp2 = nn.MaxPool2d(4, stride=4, padding=0)
        self.cv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.r2 = nn.ReLU(inplace=True)

        self.mp3 = nn.MaxPool2d(4, stride=4, padding=0)
        self.cv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.r3 = nn.ReLU(inplace=True)

        self.mp4 = nn.MaxPool2d(4, stride=4, padding=0)
        self.cv4 = nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)

        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.mp1(x)
        x = self.cv1(x)
        x = self.r1(x)
        x = self.mp2(x)
        x = self.cv2(x)
        x = self.r2(x)
        x = self.drop(x)
        x = self.mp3(x)
        x = self.cv3(x)
        x = self.r3(x)
        x = self.mp4(x)
        x = self.cv4(x)
        return x.view(-1,2)

class SimpleConfig:
    def __init__(self):
        self.batch_size = 16
        self.epochs = 5
        self.steps_per_train_loss = 10
        self.steps_per_dev_loss = 100
        self.steps_per_save = 1000

        self.cuda = True
        self.pin_memory = True

        self.checkpoint_path = 'experiments/checkpoint_binary'
        self.best_checkpoint_path = 'experiments/checkpoint_binary_best'

        self.model = BinaryNet()
        self.optimizer = Adam(self.model.parameters(), lr=.01)


    def dataloader(self, mode='train'):
        if mode == 'train':
            return DataLoader(TrainData(), 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=2)
        else:
            return DataLoader(DevData(), 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=2)


if __name__ == '__main__':
    config = SimpleConfig()
    train(config)