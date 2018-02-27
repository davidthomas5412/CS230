import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from unet.data import *
from unet.unet import UNet
from unet.loss import CrossEntropyLoss2d

import numpy as np
import argparse



def str2bool(inp):
	return inp.lower() in ('1','true')

parser = argparse.ArgumentParser()
parser.add_argument('-bilinear', type=str2bool)
args = parser.parse_args()


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


net = UNet(bilinear=args.bilinear)
net.cuda()

img_size = 512
padding = 0
batch_size = 4
criterion = CrossEntropyLoss2d(padding=padding)

optimizer = optim.Adam(net.parameters(), lr = 0.001)

checkpoint = '/scratch/users/dthomas5/exp3/trainjobs/checkpoint_bilinear_{}.txt'.format(args.bilinear)

dataset = StreakImageDataset(transform=transforms.Compose([RandomCrop(img_size, padding), RandomFlipRotate(), ToSegmentClasses(10), LogNormalize(), ToTensor()]))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

import time

for epoch in range(10):
	print('epoch {} started at {}'.format(epoch, time.time()))
	running_loss = 0.0
	for i, data in enumerate(loader):
		inputs, labels = Variable(data['inp']).cuda(), Variable(data['lab']).cuda()
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.data[0]

		if i % 10 == 9:
			print('[{},{},{}] loss: {:f}'.format(epoch + 1, i + 1, batch_size, running_loss / (10 * batch_size)))
			running_loss = 0.0

	torch.save(net.state_dict(), checkpoint)
