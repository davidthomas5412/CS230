import os

import numpy as np
from torch import from_numpy
from torch.utils.data import Dataset
from skimage.transform import rescale

class InitialData(Dataset):
	TRAIN_SIZE = 2 * 9000
	OTHER_SIZE = 100

	def __init__(self, mode='train'):
		if mode not in ['train', 'dev', 'test']:
			raise RuntimeError('Incorrect mode {}'.format(mode))

		self.mode = mode
		self.input_folder = os.path.join('data', 'initial', mode, 'input')
		self.label_folder = os.path.join('data', 'initial', mode, 'label')


	def __len__(self):
		if self.mode == 'train':
			return InitialData.TRAIN_SIZE
		else:
			return InitialData.OTHER_SIZE

	def __getitem__(self, idx):
		input_file = os.path.join(self.input_folder, '{}.npy'.format(idx))
		label_file = os.path.join(self.label_folder, '{}.npy'.format(idx))
		inp = np.load(input_file).astype('float32')
		lab = np.load(label_file).astype('float32')
		inp = inp.reshape(-1, inp.shape[0], inp.shape[1])
		lab = lab.reshape(-1, lab.shape[0], lab.shape[1])

		output = {'input': from_numpy(inp), 
				  'label': from_numpy(lab)}
		return output

class FourthDataScaled(InitialData):
	def __init__(self, mode='train'):
		InitialData.__init__(self, mode=mode)
		self.input_folder = os.path.join('data', 'fourth', mode, 'input')
		self.label_folder = os.path.join('data', 'fourth', mode, 'label')

	def __getitem__(self, idx):
		input_file = os.path.join(self.input_folder, '{}.npy'.format(idx))
		label_file = os.path.join(self.label_folder, '{}.npy'.format(idx))
		inp = np.load(input_file).astype('float32') * np.log(10 ** 5)
		lab = np.load(label_file).astype('float32') * np.log(10 ** 5)
		inp = inp.reshape(-1, inp.shape[0], inp.shape[1])
		lab = lab.reshape(-1, lab.shape[0], lab.shape[1])

		output = {'input': from_numpy(inp), 
				  'label': from_numpy(lab)}
		return output

class FourthDataScaledTest(InitialData):
	def __init__(self, mode='train'):
		InitialData.__init__(self, mode=mode)
		self.input_folder = os.path.join('data', 'fifth', mode, 'input')
		self.label_folder = os.path.join('data', 'fifth', mode, 'label')

	def __len__(self):
		return 10

	def __getitem__(self, idx):
		input_file = os.path.join(self.input_folder, '{}.npy'.format(idx))
		label_file = os.path.join(self.label_folder, '{}.npy'.format(idx))
		inp = np.load(input_file).astype('float32') * np.log(10 ** 5)
		lab = np.load(label_file).astype('float32') * np.log(10 ** 5)
		inp = inp.reshape(-1, inp.shape[0], inp.shape[1])
		lab = lab.reshape(-1, lab.shape[0], lab.shape[1])

		output = {'input': from_numpy(inp), 
				  'label': from_numpy(lab)}
		return output

class FourthData(InitialData):
	def __init__(self, mode='train'):
		InitialData.__init__(self, mode=mode)
		self.input_folder = os.path.join('data', 'fourth', mode, 'input')
		self.label_folder = os.path.join('data', 'fourth', mode, 'label')

class RescaleFourthData(InitialData):
	TRAIN_SIZE = 2 * 9000
	OTHER_SIZE = 100

	def __init__(self, mode='train'):
		if mode not in ['train', 'dev', 'test']:
			raise RuntimeError('Incorrect mode {}'.format(mode))

		self.mode = mode
		self.input_folder = os.path.join('data', 'fourth', mode, 'input')
		self.label_folder = os.path.join('data', 'fourth', mode, 'label')

	def __len__(self):
		if self.mode == 'train':
			return InitialData.TRAIN_SIZE
		else:
			return InitialData.OTHER_SIZE

	def __getitem__(self, idx):
		input_file = os.path.join(self.input_folder, '{}.npy'.format(idx))
		label_file = os.path.join(self.label_folder, '{}.npy'.format(idx))
		inp = np.load(input_file).astype('double')# * 11
		lab = np.load(label_file).astype('double')# * 11
		inp = rescale(inp, 1/4.0, order=1)
		lab = rescale(lab, 1/4.0, order=1)
		inp = inp.reshape(-1, inp.shape[0], inp.shape[1])
		lab = lab.reshape(-1, lab.shape[0], lab.shape[1])

		output = {'input': from_numpy(inp.astype('float32')),
				  'label': from_numpy(lab.astype('float32'))}
		return output

class FourthRescaleLocalTestData(Dataset):
	def __init__(self):
		self.input_folder = os.path.join('data', 'fifth', 'train', 'input')
		self.label_folder = os.path.join('data', 'fifth', 'train', 'label')

	def __len__(self):
		return 11

	def __getitem__(self, idx):
		input_file = os.path.join(self.input_folder, '{}.npy'.format(idx))
		label_file = os.path.join(self.label_folder, '{}.npy'.format(idx))
		inp = np.load(input_file).astype('double') * 11
		lab = np.load(label_file).astype('double') * 11
		inp = rescale(inp, 1/4.0, order=1)
		lab = rescale(lab, 1/4.0, order=1)
		inp = inp.reshape(-1, inp.shape[0], inp.shape[1])
		lab = lab.reshape(-1, lab.shape[0], lab.shape[1])

		output = {'input': from_numpy(inp.astype('float32')), 
				  'label': from_numpy(lab.astype('float32'))}
		return output

class FourthLocalTestData(Dataset):
	def __init__(self):
		self.input_folder = os.path.join('data', 'fifth', 'train', 'input')
		self.label_folder = os.path.join('data', 'fifth', 'train', 'label')

	def __len__(self):
		return 11

	def __getitem__(self, idx):
		input_file = os.path.join(self.input_folder, '{}.npy'.format(idx))
		label_file = os.path.join(self.label_folder, '{}.npy'.format(idx))
		inp = np.load(input_file).astype('float32')
		lab = np.load(label_file).astype('float32')
		inp = inp.reshape(-1, inp.shape[0], inp.shape[1])
		lab = lab.reshape(-1, lab.shape[0], lab.shape[1])

		output = {'input': from_numpy(inp), 
				  'label': from_numpy(lab)}
		return output

