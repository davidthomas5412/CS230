import os
import torch

import numpy as np
import torch.optim as optim
from torch.autograd import Variable

from time import time

def train(config):
	np.random.seed(0)
	torch.manual_seed(0)
	if config.cuda:
		torch.cuda.manual_seed(0)

	model = config.model.train()
	if config.cuda:
		model = model.cuda()
	loss_fn = config.loss_fn
	optimizer = config.optimizer

	for epoch in range(config.epochs):

		for i, sample in enumerate(config.dataloader(mode='train')):
			input_batch = sample['input']
			label_batch = sample['label']

			# move to GPU, convert to Variables
			if config.cuda:
				input_batch = input_batch.cuda(async=True)
				label_batch = label_batch.cuda(async=True)

			input_batch = Variable(input_batch)
			label_batch = Variable(label_batch)

			# forward pass
			output_batch = model(input_batch)
			train_loss = loss_fn(output_batch, label_batch)

			# backward pass
			optimizer.zero_grad()
			train_loss.backward()
			optimizer.step()

			if i % config.steps_per_train_loss == 0:
				train_loss_val = np.asscalar(train_loss.data.cpu().numpy())
				message = '[{}, {}] train_loss: {}'.format(epoch, i, train_loss_val)
				print(message)

			if i % config.steps_per_dev_loss == 0:
				print('dev loss time', time())
				dev_loss_vals = []
				for j, dev_sample in enumerate(config.dataloader(mode='dev')):
					print(j)
					input_batch = dev_sample['input']
					label_batch = dev_sample['label']

					# move to GPU, convert to Variables
					if config.cuda:
						input_batch = input_batch.cuda(async=True)
						label_batch = label_batch.cuda(async=True)

					input_batch = Variable(input_batch)
					label_batch = Variable(label_batch)
					
					output_batch = model(input_batch)
					dev_loss = loss_fn(output_batch.cpu(), label_batch.cpu())
					dev_loss_val = np.asscalar(dev_loss.data.cpu().numpy())
					dev_loss_vals.append(dev_loss_val)
				message = '[{}, {}] avg_dev_loss: {}'.format(epoch, i, np.mean(dev_loss_vals))
				print(message)

		# write checkpoint after each epoch
		state = model.state_dict()
		torch.save(state, config.checkpoint_path)


if __name__ == '__main__':
	from model.experiment import *
	config = FourthBNDataConfig()
	train(config)