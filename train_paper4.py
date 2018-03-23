import os
import torch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn import MSELoss, L1Loss
from torch.utils.data import Dataset, DataLoader

from time import time

"""
To test whether getting rid of edges will help.
"""

def toscalar(var):
	return np.mean(var.data.cpu().numpy())

def train(config):
	np.random.seed(0)
	torch.manual_seed(0)
	if config.cuda:
		torch.cuda.manual_seed(0)
		torch.cuda.set_device(1)

	model = config.model
	model.load_state_dict(torch.load('experiments/checkpoint_paper2'))
	model.train()
	model.cuda()
	optimizer = config.optimizer

	mse = MSELoss()
	l1loss = L1Loss()
	best_loss = 10

	for epoch in range(config.epochs):

		for i, sample in enumerate(config.dataloader(mode='train')):
			input_batch = Variable(sample['input'].cuda(async=True))
			label_batch = Variable(sample['label'].cuda(async=True))

			# forward pass
			output_batch = model(input_batch)

			loss = 0.05 * l1loss(output_batch, label_batch) + mse(output_batch[:,:,40:-40,40:-40], label_batch[:,:,40:-40,40:-40])

			# backward pass
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i % config.steps_per_train_loss == 0:
				message = '[{}, {}] mse: {}'.format(epoch, i, toscalar(loss))
				print(message)

			if i % config.steps_per_dev_loss == 0:
				model.eval()
				print('time', time())
				dloss_arr = []

				for j, dev_sample in enumerate(config.dataloader(mode='dev')):
					input_batch = Variable(dev_sample['input'].cuda(async=True))
					label_batch = Variable(dev_sample['label'].cuda(async=True))
					
					output_batch = model(input_batch)

					dloss_arr.append(toscalar(mse(output_batch[:,:,40:-40,40:-40], label_batch[:,:,40:-40,40:-40])))
				dloss = np.mean(dloss_arr)

				if dloss < best_loss:
					print('best')
					best_loss = dloss
					state = model.state_dict()
					torch.save(state, config.best_checkpoint_path)

				message = '[dev] mmse: {}'.format(np.mean(dloss))
				print(message)
				model.train()

			if i % config.steps_per_save == 0:
				state = model.state_dict()
				torch.save(state, config.checkpoint_path)

		# write checkpoint after each epoch
		state = model.state_dict()
		torch.save(state, config.checkpoint_path)

class AugmentedData(Dataset):

	TRAIN_SIZE = 2 * 90000
	OTHER_SIZE = 100

	def __init__(self, mode='train'):
		if mode not in ['train', 'dev', 'test']:
			raise RuntimeError('Incorrect mode {}'.format(mode))

		self.mode = mode
		self.input_folder = os.path.join('data', 'fifth', mode, 'input')
		self.label_folder = os.path.join('data', 'fifth', mode, 'label')

	def __len__(self):
		if self.mode == 'train':
			return AugmentedData.TRAIN_SIZE
		else:
			return AugmentedData.OTHER_SIZE

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
		self.batch_size = 16
		self.epochs = 5
		self.steps_per_train_loss = 20
		self.steps_per_dev_loss = 100
		self.steps_per_save = 500

		self.cuda = True
		self.pin_memory = True

		self.checkpoint_path = 'experiments/checkpoint_paper4'
		self.best_checkpoint_path = 'experiments/checkpoint_paper4_best'


		self.model = Net()
		self.optimizer = Adam(self.model.parameters(), lr=.01)


	def dataloader(self, mode='train'):
		if mode == 'train':
			return DataLoader(AugmentedData(mode=mode), 
				batch_size=self.batch_size, 
				shuffle=True,
        		num_workers=2,
        		pin_memory=self.pin_memory)
		else:
			return DataLoader(AugmentedData(mode=mode), 
				batch_size=16, 
				shuffle=False,
        		num_workers=2,
        		pin_memory=self.pin_memory)


if __name__ == '__main__':
	config = SimpleConfig()
	train(config)