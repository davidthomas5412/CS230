import numpy as np
from scipy.ndimage.interpolation import rotate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from astropy.io import fits

class StreakImageDataset(Dataset):
	"""Dataset for LSST streak image project."""
	def __init__(self, transform=None, is_train=True):
		self.transform = transform
		self.backgrounds = 90
		self.masks = 10
		self.is_train = is_train

	def __len__(self):
		if self.is_train:
			return self.backgrounds * self.masks * 2
		return (100 - self.backgrounds) * self.masks * 2 # TODO: make this cleaner.

	def __getitem__(self, idx):
		if not self.is_train:
			idx += self.backgrounds * self.masks * 2 # TODO: clean this up
		
		back_idx = idx // 20
		var_idx = idx % 2
		mask_idx = (idx % 20) // 2

		if not self.is_train:
			back_idx += self.backgrounds

		backf = 'lsst_e_{}_f2_R22_S11_E000.fits'.format(back_idx)
		back = fits.open('/scratch/users/dthomas5/exp3/base/output/{}'.format(backf))[0].data
		if var_idx:
			#nonvariable
			inp = fits.open('/scratch/users/dthomas5/exp3/variable1/mask{}/output/{}'.format(mask_idx, backf))[0].data
			lab = np.zeros(inp.shape)
		else:
			#variable
			inp = fits.open('/scratch/users/dthomas5/exp3/variable2/mask{}/output/{}'.format(mask_idx, backf))[0].data
			lab = fits.open('/scratch/users/dthomas5/exp3/variable3/mask{}/output/{}'.format(mask_idx, backf))[0].data
		inp += back

		sample = {'inp': inp, 'lab': lab}
		if self.transform:
			sample = self.transform(sample)

		return sample

class RandomFlipRotate:
	def __init__(self):
		self.angles = [0, 90, 180, 270]

	def __call__(self, sample):
		inp, lab = sample['inp'], sample['lab']
		if np.random.uniform() > 0.5:
			inp = np.fliplr(inp)
			lab = np.fliplr(lab)
		rot = np.random.randint(4)
		inp = rotate(inp, self.angles[rot], order=1)
		lab = rotate(lab, self.angles[rot], order=1)
		return {'inp': inp, 'lab': lab}


class RandomCrop:
	def __init__(self, img_size, padding):
		self.img_size = img_size
		self.padding = padding

	def __call__(self, sample):
		big_inp, big_lab = sample['inp'], sample['lab']
		x = np.random.randint(0, big_inp.shape[0] - self.img_size)
		y = np.random.randint(0, big_inp.shape[1] - self.img_size)
		inp = big_inp[x:(x + self.img_size), y:(y + self.img_size)]
		lab = big_lab[(x + self.padding): (x + self.img_size - self.padding),
					  (y + self.padding): (y + self.img_size - self.padding)]
		return {'inp': inp, 'lab': lab}


class LogNormalize:
	def __call__(self, sample):
		inp, lab = sample['inp'], sample['lab']
		inp = np.log(inp + 1)
		return {'inp': inp, 'lab': lab}

class ToSegmentClasses:
	def __init__(self, cutoff):
		self.cutoff = cutoff

	def __call__(self, sample):
		inp, lab = sample['inp'], sample['lab']
		lab = lab > self.cutoff
		return {'inp': inp, 'lab': lab}

class ToTensor:
	def __call__(self, sample):
		inp, lab = sample['inp'], sample['lab']
		# swap color axis because
		# numpy image: 400 x 400 x 3
		# torch image: 400 X 400 X 3
		inp = inp.reshape(-1, inp.shape[0], inp.shape[1])
		lab = lab.reshape(lab.shape[0], lab.shape[1]).astype('int64')
		return {'inp': torch.from_numpy(inp), 'lab': torch.from_numpy(lab)}

if __name__ == '__main__':
	np.random.seed(1)
	torch.manual_seed(1)

	dataset = StreakImageDataset(transform=transforms.Compose([RandomCrop(400), RandomFlipRotate(), ToSegmentClasses(), ToTensor()]))
	

	sample1 = dataset[10]
	sample2 = dataset[11]
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	from matplotlib.colors import LogNorm

	plt.figure()
	plt.imshow(sample1['inp'].numpy()[0,:,:], norm=LogNorm(), cmap='hot')
	plt.savefig('foo1.png')

	plt.figure()
	plt.imshow(sample1['lab'].numpy()[0,:,:] > 10, cmap='hot')
	plt.savefig('foo2.png')

	plt.figure()
	plt.imshow(sample2['inp'].numpy()[0,:,:], norm=LogNorm(), cmap='hot')
	plt.savefig('foo3.png')

	plt.figure()
	plt.imshow(sample2['lab'].numpy()[0,:,:] > 10, cmap='hot')
	plt.savefig('foo4.png')