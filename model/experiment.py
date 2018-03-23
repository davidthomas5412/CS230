import torch
import numpy as np
from model.config import Config
from model.simplenet import *
from model.data import *
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from model.loss import WeightedL1Loss


class SimpleConfig(Config):
	def __init__(self):
		super(L2PenaltyConfig).__init__()
		self.batch_size = 16
		self.epochs = 3
		self.steps_per_train_loss = 10
		self.steps_per_dev_loss = 100

		self.cuda = True
		self.pin_memory = True

		self.checkpoint_path = 'experiments/checkpoint_simple'

		self.model = SimpleBatchnormNet()
		self.optimizer = Adam(self.model.parameters(), lr=.01)
		self.loss_fn = MSELoss()

	def dataloader(self, mode='train'):
		if mode == 'train':
			return DataLoader(FourthDataScaled(mode=mode), 
				batch_size=self.batch_size, 
				shuffle=True,
        		num_workers=1,
        		pin_memory=self.pin_memory)
		else:
			return DataLoader(FourthDataScaled(mode=mode), 
				batch_size=2, 
				shuffle=False,
        		num_workers=1,
        		pin_memory=self.pin_memory)

class DownsampledConfig(Config):
	def __init__(self):
		super(L2PenaltyConfig).__init__()
		self.batch_size = 16
		self.epochs = 1
		self.steps_per_train_loss = 10
		self.steps_per_dev_loss = 100

		self.cuda = True
		self.pin_memory = True

		self.checkpoint_path = 'experiments/batchnormagain/checkpoint_downsampled5_multloss'

		self.model = SimpleBatchnormNet()
		self.optimizer = Adam(self.model.parameters(), lr=.01)
		self.loss_fn = MSELoss()

	def dataloader(self, mode='train'):
		if mode == 'train':
			return DataLoader(RescaleFourthData(mode=mode), 
				batch_size=self.batch_size, 
				shuffle=True,
        		num_workers=1,
        		pin_memory=self.pin_memory)
		else:
			return DataLoader(RescaleFourthData(mode=mode), 
				batch_size=2, 
				shuffle=False,
        		num_workers=1,
        		pin_memory=self.pin_memory)

class FourthBNDataConfig(Config):
	def __init__(self):
		super(L2PenaltyConfig).__init__()
		self.batch_size = 16
		self.epochs = 1	
		self.steps_per_train_loss = 10
		self.steps_per_dev_loss = 100

		self.cuda = True
		self.pin_memory = True

		self.checkpoint_path = 'experiments/batchnormagain/checkpoint_bn'

		self.model = DeeperBNNet()
		self.optimizer = Adam(self.model.parameters(), lr=.01)
		self.loss_fn = MSELoss()

	def dataloader(self, mode='train'):
		if mode == 'train':
			return DataLoader(FourthData(mode=mode), 
				batch_size=self.batch_size, 
				shuffle=True,
        		num_workers=1,
        		pin_memory=self.pin_memory)
		else:
			return DataLoader(FourthData(mode=mode), 
				batch_size=2, 
				shuffle=False,
        		num_workers=1,
        		pin_memory=self.pin_memory)


class HalfConfig(FourthBNDataConfig):
	def __init__(self):
		FourthBNDataConfig.__init__(self)
		self.checkpoint_path = 'experiments/batchnormagain/checkpoint_nobn'
		self.model = HalfNet()
		self.optimizer = Adam(self.model.parameters(), lr=0.01)

class SevenConvConfig(FourthBNDataConfig):
	def __init__(self):
		FourthBNDataConfig.__init__(self)
		self.checkpoint_path = 'experiments/batchnormagain/checkpoint_nobn'
		self.model = SevenConvNet()
		self.optimizer = Adam(self.model.parameters(), lr=0.01)

class FiveConvConfig(FourthBNDataConfig):
	def __init__(self):
		FourthBNDataConfig.__init__(self)
		self.checkpoint_path = 'experiments/batchnormagain/checkpoint_nobn'
		self.model = FiveConvNet()
		self.optimizer = Adam(self.model.parameters(), lr=0.01)

class LRConfig(FourthBNDataConfig):
	def __init__(self, lr):
		FourthBNDataConfig.__init__(self)
		self.checkpoint_path = 'experiments/batchnormagain/checkpoint_nobn'
		self.model = DeeperNet()
		self.optimizer = Adam(self.model.parameters(), lr=lr)

class BNConfig(FourthBNDataConfig):
	def __init__(self):
		FourthBNDataConfig.__init__(self)
		self.checkpoint_path = 'experiments/batchnormagain/checkpoint_nobn'
		self.model = DeeperBNNet()
		self.optimizer = Adam(self.model.parameters(), lr=.01)

class NoBNConfig(FourthBNDataConfig):
	def __init__(self):
		FourthBNDataConfig.__init__(self)
		self.checkpoint_path = 'experiments/batchnormagain/checkpoint_nobn'
		self.model = DeeperNet()
		self.optimizer = Adam(self.model.parameters(), lr=.01)


class L2PenaltyConfig(Config):
	def __init__(self):
		super(L2PenaltyConfig).__init__()
		self.batch_size = 16
		self.epochs = 20
		self.steps_per_train_loss = 20
		self.steps_per_dev_loss = 200

		self.cuda = True
		self.pin_memory = True

		self.checkpoint_path = 'experiments/l2loss/checkpoint'

		self.model = SimpleBatchnormNet()
		self.optimizer = Adam(self.model.parameters(), lr=.01)
		self.loss_fn = MSELoss()

		# np.random.seed(0)
		# torch.manual_seed(0)

	def dataloader(self, mode='train'):
		if mode == 'train':
			return DataLoader(InitialData(mode=mode), 
				batch_size=self.batch_size, 
				shuffle=True,
        		num_workers=1,
        		pin_memory=self.pin_memory)
		else:
			return DataLoader(InitialData(mode=mode), 
				batch_size=self.batch_size, 
				shuffle=False,
        		num_workers=1,
        		pin_memory=self.pin_memory)

class WeightedL1PenaltyConfig(L2PenaltyConfig):
	def __init__(self):
		L2PenaltyConfig.__init__(self)

		self.checkpoint_path = 'experiments/weightedl1loss/checkpoint'

		self.loss_fn = WeightedL1Loss(weight=100.0)

class DeeperModelConfig(Config):
	def __init__(self):
		super(WithBatchnormConfig).__init__()
		self.batch_size = 16
		self.epochs = 60
		self.steps_per_train_loss = 20
		self.steps_per_dev_loss = 200

		self.cuda = True
		self.pin_memory = True

		self.checkpoint_path = 'experiments/deepermodel/deeper_model_checkpoint'

		self.model = DeeperNet()
		self.optimizer = Adam(self.model.parameters())
		self.loss_fn = L1Loss()

		# np.random.seed(0)
		# torch.manual_seed(0)

	def dataloader(self, mode='train'):
		if mode == 'train':
			return DataLoader(InitialData(mode=mode), 
				batch_size=self.batch_size, 
				shuffle=True,
        		num_workers=1,
        		pin_memory=self.pin_memory)
		else:
			return DataLoader(InitialData(mode=mode), 
				batch_size=self.batch_size, 
				shuffle=False,
        		num_workers=1,
        		pin_memory=self.pin_memory)


class WithBatchnormConfig(Config):
	def __init__(self):
		super(WithBatchnormConfig).__init__()
		self.batch_size = 16
		self.epochs = 4
		self.steps_per_train_loss = 20
		self.steps_per_dev_loss = 200


		self.cuda = True
		self.pin_memory = True

		self.checkpoint_path = 'experiments/addbatchnorm/high_learning_rate_long_training_checkpoint'

		self.model = SimpleBatchnormNet()
		self.optimizer = Adam(self.model.parameters(), lr=0.01)
		self.loss_fn = L1Loss()

		# np.random.seed(0)
		# torch.manual_seed(0)

	def dataloader(self, mode='train'):
		if mode == 'train':
			return DataLoader(InitialData(mode=mode), 
				batch_size=self.batch_size, 
				shuffle=True,
        		num_workers=1,
        		pin_memory=self.pin_memory)
		else:
			return DataLoader(InitialData(mode=mode), 
				batch_size=self.batch_size, 
				shuffle=False,
        		num_workers=1,
        		pin_memory=self.pin_memory)

class FirstConfig(Config):
	def __init__(self):
		super(FirstConfig).__init__()
		self.batch_size = 16
		self.epochs = 1
		self.steps_per_train_loss = 20
		self.steps_per_dev_loss = 200


		self.cuda = True
		self.pin_memory = True

		self.checkpoint_path = 'experiments/first/checkpoint'

		self.model = SimpleNet()
		self.optimizer = Adam(self.model.parameters())
		self.loss_fn = L1Loss()

		# np.random.seed(0)
		# torch.manual_seed(0)

	def dataloader(self, mode='train'):
		if mode == 'train':
			return DataLoader(InitialData(mode=mode), 
				batch_size=self.batch_size, 
				shuffle=True,
        		num_workers=1,
        		pin_memory=self.pin_memory)
		else:
			return DataLoader(InitialData(mode=mode), 
				batch_size=self.batch_size, 
				shuffle=False,
        		num_workers=1,
        		pin_memory=self.pin_memory)

if __name__ == '__main__':
	print(WeightedL1PenaltyConfig().cuda)
