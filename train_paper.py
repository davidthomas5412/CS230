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
	np.random.seed(0)
	torch.manual_seed(0)
	if config.cuda:
		torch.cuda.manual_seed(0)

	model = config.model
	model.train()
	model.cuda()
	optimizer = config.optimizer

	mse = MSELoss()
	nllloss = NLLLoss()
	fploss = FalsePositive()

	acc = Accuracy()

	for epoch in range(config.epochs):

		for i, sample in enumerate(config.dataloader(mode='train')):
			input_batch = Variable(sample['input'].cuda(async=True))
			img_batch = Variable(sample['img'].cuda(async=True))
			label_batch = Variable(sample['label'].cuda(async=True))

			# forward pass
			output_batch, pred_batch = model(input_batch)

			loss1 = mse(output_batch, img_batch)
			loss2 = 0.01 * nllloss(pred_batch, label_batch)
			loss3 = fploss(pred_batch, label_batch)
			loss = loss1 + loss2 + loss3

			# backward pass
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i % config.steps_per_train_loss == 0:
				message = '[{}, {}] mse: {} nllloss: {} fploss: {}'.format(epoch, i,
					toscalar(loss1), toscalar(loss2), toscalar(loss3))
				print(message)

			if i % config.steps_per_dev_loss == 0:
				model.eval()
				print('time', time())
				dloss1_arr = []
				dloss2_arr = []
				dloss3_arr = []
				acc_sum = 0

				for j, dev_sample in enumerate(config.dataloader(mode='dev')):
					input_batch = Variable(dev_sample['input'].cuda(async=True))
					img_batch = Variable(dev_sample['img'].cuda(async=True))
					label_batch = Variable(dev_sample['label'].cuda(async=True))
					
					output_batch, pred_batch = model(input_batch)

					dloss1_arr.append(toscalar(mse(output_batch, img_batch)))
					dloss2_arr.append(toscalar(0.01 * nllloss(pred_batch, label_batch)))
					dloss3_arr.append(toscalar(fploss(pred_batch, label_batch)))
					acc_sum += toscalar(acc(pred_batch, label_batch))
					
				message = '[dev] mmse: {} mnllloss: {} mfploss: {} acc: {}'.format(
					np.mean(dloss1_arr), np.mean(dloss2_arr), np.mean(dloss3_arr), acc_sum)
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
				  'img': torch.from_numpy(lab),
				  'label': torch.from_numpy(np.array([(idx % 2 == 1)], dtype='int64')) 
				  }
		return output

class Accuracy(torch.nn.Module):
    def forward(self, x, y):
    	return torch.sum((x[:,1] > 0.5) * (y.float() > 0.5) + (x[:,1] < 0.5) * (y.float() < 0.5))

class FalsePositive(torch.nn.Module):
    def forward(self, x, y):
    	return torch.mean((((x[:,1] > 0.5) * (y.float() < 0.5))).float())

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
        pred = self.binary(img)

        return img, pred

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.binary = nn.Sequential(
            nn.AvgPool2d(8, stride=4, padding=2),
            nn.Conv2d(1, 4, 4, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 4, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 2, 4, stride=1),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x = torch.cat([orig, intermediate], dim=1)
        x = self.binary(x)
        return x.view(-1,2,1)

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
		self.epochs = 3
		self.steps_per_train_loss = 20
		self.steps_per_dev_loss = 100
		self.steps_per_save = 500

		self.cuda = True
		self.pin_memory = True

		self.checkpoint_path = 'experiments/checkpoint_paper'

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