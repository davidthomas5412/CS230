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

def toto(val):
	return np.asscalar(val.data.cpu().numpy())

def train(config):
	np.random.seed(0)
	torch.manual_seed(0)

	model = config.model
	model.train()
	optimizer = config.optimizer

	nllloss = NLLLoss()
	fploss = FalsePositive()
	fnloss = FalseNegative()

	fpcount = FalsePositiveCount()
	fncount = FalseNegativeCount()

	acc = Accuracy()

	for epoch in range(config.epochs):

		for i, sample in enumerate(config.dataloader()):
			img_batch = Variable(sample['img'])
			label_batch = Variable(sample['label'])

			# forward pass
			pred_batch = model(img_batch)
			loss = nllloss(pred_batch, label_batch)
			
			if i % 10 == 0:
				print('loss{} '.format(i), toto(0.1 * nllloss(pred_batch, label_batch)), toto(acc(pred_batch, label_batch)))

			# backward pass
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()


		# write checkpoint after each epoch
		# state = model.state_dict()
		# torch.save(state, config.checkpoint_path)

class AugmentedData(Dataset):

	def __init__(self):
		self.label_folder = os.path.join('data', 'fourth', 'train', 'label')

	def __len__(self):
		return 2600

	def __getitem__(self, idx):
		label_file = os.path.join(self.label_folder, '{}.npy'.format(idx))
		lab = np.load(label_file).astype('float32') * np.log(10 ** 5)
		lab = lab.reshape(-1, lab.shape[0], lab.shape[1])

		output = { 
				  'img': torch.from_numpy(lab),
				  'label': torch.from_numpy(np.array([(idx % 2 == 1)], dtype='int64')) 
				  }
		return output

class Accuracy(torch.nn.Module):
    def forward(self, x, y):
    	return torch.sum((x[:,1] > 0.5) * (y.float() > 0.5) + (x[:,1] < 0.5) * (y.float() < 0.5))

class FalsePositive(torch.nn.Module):
    def forward(self, x, y):
    	return torch.mean(((x[:,1] > 0.5) * (y < 1)).float())

class FalseNegative(torch.nn.Module):
    def forward(self, x, y):
    	return torch.mean(((x[:,1] < 0.5) * (y > 0)).float())

class FalsePositiveCount(torch.nn.Module):
    def forward(self, x, y):
        return torch.sum(((x[:,1] > 0.5) * (y < 1)).float())

class FalseNegativeCount(torch.nn.Module):
    def forward(self, x, y):
        return torch.sum(((x[:,0] > 0.5) * (y > 0)).float())

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

class SimpleConfig:
	def __init__(self):
		self.batch_size = 16
		self.epochs = 3
		self.steps_per_train_loss = 10
		self.steps_per_dev_loss = 100

		self.checkpoint_path = 'experiments/checkpoint_binary'

		self.model = BinaryClassifier()
		self.optimizer = Adam(self.model.parameters(), lr=.01)


	def dataloader(self):
		return DataLoader(AugmentedData(), 
				batch_size=self.batch_size, 
				shuffle=True,
        		num_workers=2)


if __name__ == '__main__':
	config = SimpleConfig()
	train(config)