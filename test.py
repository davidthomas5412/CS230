import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from unet.data import *
from unet.segnet import SegNet
from unet.loss import CrossEntropyLoss2d

import numpy as np

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

net = SegNet(bilinear=True).eval()
net.cuda()
net.load_state_dict(torch.load('/scratch/users/dthomas5/exp3/trainjobs/checkpoint_bilinear_True.txt'))

img_size = 512
padding = 0
batch_size=1
criterion = CrossEntropyLoss2d(padding=padding)


dataset = StreakImageDataset(transform=transforms.Compose([RandomCrop(img_size, padding), 
	RandomFlipRotate(), ToSegmentClasses(10), LogNormalize(), ToTensor()]), is_train=False)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

running_loss = 0.0
for i, data in enumerate(loader):
	inputs, labels = Variable(data['inp']).cuda(), Variable(data['lab']).cuda()
	outputs = net(inputs)
	loss = criterion(outputs, labels)
	running_loss += loss.data[0]
	print(i, ' - ', loss.data[0])
	if i < 100:
		np.save('/scratch/users/dthomas5/exp3/testimgs2/input{}'.format(i), inputs.cpu().data.numpy())
		np.save('/scratch/users/dthomas5/exp3/testimgs2/output{}'.format(i), outputs.cpu().data.numpy())
		np.save('/scratch/users/dthomas5/exp3/testimgs2/label{}'.format(i), labels.cpu().data.numpy())
	else:
		break

print('test loss [1,100,1]: {:f}'.format(running_loss))

