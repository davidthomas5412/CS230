import torch
import numpy as np

from torch.utils.data import DataLoader
from model.data import *
from model.simplenet import SimpleBatchnormNet
from torch.autograd import Variable


dn = SimpleBatchnormNet()
dn.load_state_dict(torch.load('experiments/checkpoint_simple',
		map_location=lambda storage, loc:storage))
dn = dn.eval()

loader = DataLoader(FourthDataScaledTest(), batch_size=1, num_workers=1)


for i,sample in enumerate(loader):

	input_batch = Variable(sample['input'])
	label_batch = Variable(sample['label'])
	output_batch = dn(input_batch)

	np.save('/Users/user/Code/Astronomy/exp4/notebooks/test/first/i{}.npy'.format(i),
		input_batch.data.numpy())
	np.save('/Users/user/Code/Astronomy/exp4/notebooks/test/first/l{}.npy'.format(i),
		label_batch.data.numpy())
	np.save('/Users/user/Code/Astronomy/exp4/notebooks/test/first/o{}.npy'.format(i),
		output_batch.data.numpy())