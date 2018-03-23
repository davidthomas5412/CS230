import os

import numpy as np
from astropy.io import fits
from skimage.transform import rescale


def build_durtest_dataset():
    # fix random seed for reproducibility
    np.random.seed(10)

    # make directories
    base = os.path.join('data', 'durtest')
    if not os.path.exists(base):
        os.mkdir(base)

    smart_crop = SmartCrop(size=512)
    var_transforms = [smart_crop, Downsample(scale=(1 / 2.0)), RotateFlip(), Normalize2(maxval=10**5)]
    
    copy_crop = CopyCrop(smart_crop, size=512)
    nonvar_transforms = [copy_crop, Downsample(scale=(1 / 2.0)), RotateFlip(), Normalize2(maxval=10**5)]
    def apply_transforms(transforms, sample_dict):
        for t in transforms:
            sample_dict = t.transform(sample_dict)
        return sample_dict

    data = AstroTestData()

    new_id = 0

    def make(inp, lab, i_back, inp_dir, lab_dir, var):
        sample_dict = {
            'input': inp,
            'label': lab
        }
        if var:
            sample_dict = apply_transforms(var_transforms, sample_dict)
        else:
            sample_dict = apply_transforms(nonvar_transforms, sample_dict)
        input_path = os.path.join(inp_dir, str(i_back))
        np.save(input_path, sample_dict['input'])
        label_path = os.path.join(lab_dir, str(i_back))
        np.save(label_path, sample_dict['label'])
    for dur in [0.000316, 0.000513, 0.000832, 0.001350, 0.002189, 0.003551, 0.005759, 0.009342, 0.015153, 0.024579, 0.200000]:
        print('dur', dur)
        dur_dir = os.path.join(base, str(dur))
        var_dir = os.path.join(dur_dir, 'var')
        nonvar_dir = os.path.join(dur_dir, 'nonvar')
        inp_var_dir = os.path.join(var_dir, 'inp')
        lab_var_dir = os.path.join(var_dir, 'lab')
        inp_nonvar_dir = os.path.join(nonvar_dir, 'inp')
        lab_nonvar_dir = os.path.join(nonvar_dir, 'lab')
        if not os.path.exists(dur_dir):
            os.mkdir(dur_dir)
            os.mkdir(var_dir)
            os.mkdir(nonvar_dir) 
            os.mkdir(inp_var_dir)
            os.mkdir(lab_var_dir) 
            os.mkdir(inp_nonvar_dir)
            os.mkdir(lab_nonvar_dir)

        new_id = 0
        for background_id in range(90, 100):
            back = data.get_background(background_id)
            for _ in range(1): # different crops
                height = 10.0
                mag = 14.0

                v1 = data.get_mask(background_id, mag, height, dur, variable=1)
                v2 = data.get_mask(background_id, mag, height, dur, variable=2)
                v3 = data.get_mask(background_id, mag, height, dur, variable=3)
                zeros = np.zeros(v1.shape)
    
                make(back + v2, v3, new_id, inp_var_dir, lab_var_dir, True)
                make(back + v1, zeros, new_id, inp_nonvar_dir, lab_nonvar_dir, False)
                new_id += 1

def build_magtest_dataset():
    # fix random seed for reproducibility
    np.random.seed(10)

    # make directories
    base = os.path.join('data', 'magtest')
    if not os.path.exists(base):
        os.mkdir(base)

    smart_crop = SmartCrop(size=512)
    var_transforms = [smart_crop, Downsample(scale=(1 / 2.0)), RotateFlip(), Normalize2(maxval=10**5)]
    
    copy_crop = CopyCrop(smart_crop, size=512)
    nonvar_transforms = [copy_crop, Downsample(scale=(1 / 2.0)), RotateFlip(), Normalize2(maxval=10**5)]
    def apply_transforms(transforms, sample_dict):
        for t in transforms:
            sample_dict = t.transform(sample_dict)
        return sample_dict

    data = AstroTestData()

    new_id = 0

    def make(inp, lab, i_back, inp_dir, lab_dir, var):
        sample_dict = {
            'input': inp,
            'label': lab
        }
        if var:
            sample_dict = apply_transforms(var_transforms, sample_dict)
        else:
            sample_dict = apply_transforms(nonvar_transforms, sample_dict)
        input_path = os.path.join(inp_dir, str(i_back))
        np.save(input_path, sample_dict['input'])
        label_path = os.path.join(lab_dir, str(i_back))
        np.save(label_path, sample_dict['label'])
    
    for mag in np.linspace(14,18,9):
        print('mag', mag)
        mag_dir = os.path.join(base, str(mag))
        var_dir = os.path.join(mag_dir, 'var')
        nonvar_dir = os.path.join(mag_dir, 'nonvar')
        inp_var_dir = os.path.join(var_dir, 'inp')
        lab_var_dir = os.path.join(var_dir, 'lab')
        inp_nonvar_dir = os.path.join(nonvar_dir, 'inp')
        lab_nonvar_dir = os.path.join(nonvar_dir, 'lab')
        if not os.path.exists(mag_dir):
            os.mkdir(mag_dir)
            os.mkdir(var_dir)
            os.mkdir(nonvar_dir) 
            os.mkdir(inp_var_dir)
            os.mkdir(lab_var_dir) 
            os.mkdir(inp_nonvar_dir)
            os.mkdir(lab_nonvar_dir)

        new_id = 0
        for background_id in range(90, 100):
            back = data.get_background(background_id)
            for _ in range(1): # different crops
                height = 10.0
                dur = 0.2
                v1 = data.get_mask(background_id, mag, height, dur, variable=1)
                v2 = data.get_mask(background_id, mag, height, dur, variable=2)
                v3 = data.get_mask(background_id, mag, height, dur, variable=3)
                zeros = np.zeros(v1.shape)
    
                make(back + v2, v3, new_id, inp_var_dir, lab_var_dir, True)
                make(back + v1, zeros, new_id, inp_nonvar_dir, lab_nonvar_dir, False)
                new_id += 1

class AstroTestData:
    DIR = '/scratch/users/dthomas5/exp3/test/jobs'
    SHAPE = (4072, 4000)

    def get_background(self, background_id):
        file = os.path.join(SherlockData.EXP3,
            'base',
            'output',
            'lsst_e_{}_f2_R22_S11_E000.fits'.format(background_id))
        back = fits.open(file)[0].data
        return back

    def get_mask(self, background_id, mag, height, dur, variable=1):
        file = os.path.join(AstroTestData.DIR, 
            'mag_{:.1f}_height_{:.1f}_dur_{:.6f}_id_{:d}_v_{:d}'.format(mag, height, dur, background_id, variable),
            'lsst_e_{}_f2_R22_S11_E000.fits.gz'.format(background_id))
        mask = fits.open(file)[0].data
        return mask


# 180,000 samples
def build_fifth_dataset():
    # fix random seed for reproducibility
    np.random.seed(10)

    # make directories
    base = os.path.join('data', 'fourth')
    if not os.path.exists(base):
        os.mkdir(base)
    for folder in ['train', 'dev', 'test']:
        path = os.path.join(base, folder)
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(os.path.join(path, 'input'))
            os.mkdir(os.path.join(path, 'label'))

    transforms = [RandomCrop(size=512), Downsample(scale=(1 / 2.0)), RotateFlip(), Normalize2(maxval=10**5)]

    def apply_transforms(sample_dict):
        for t in transforms:
            sample_dict = t.transform(sample_dict)
        return sample_dict

    fast = SherlockDataFast()

    new_id = 0

    def make(inp, lab, new_id, traindevtest='train'):
        sample_dict = {
            'input': inp,
            'label': lab
        }
        sample_dict = apply_transforms(sample_dict)
        input_path = os.path.join(base, traindevtest, 'input', str(new_id))
        np.save(input_path, sample_dict['input'])
        label_path = os.path.join(base, traindevtest, 'label', str(new_id))
        np.save(label_path, sample_dict['label'])
    
    for background_id in range(90):
        back = fast.get_background(background_id)
        for mask_id in range(10):
            v1 = fast.get_mask(background_id, mask_id, variable=1)
            v2 = fast.get_mask(background_id, mask_id, variable=2)
            v3 = fast.get_mask(background_id, mask_id, variable=3)
            zeros = np.zeros(v1.shape)
            for _ in range(100): # different crops, flips, rotations
                make(back + v2, v3, new_id, traindevtest='train') # variable
                new_id += 1
                make(back + v1, zeros, new_id, traindevtest='train') # nonvariable
                new_id += 1

    new_id = 0
    for background_id in range(90, 100):
        back = fast.get_background(background_id)
        for mask_id in range(10):
            v1 = fast.get_mask(background_id, mask_id, variable=1)
            v2 = fast.get_mask(background_id, mask_id, variable=2)
            v3 = fast.get_mask(background_id, mask_id, variable=3)
            zeros = np.zeros(v1.shape)

            make(back + v2, v3, new_id, traindevtest='dev')
            make(back + v2, v3, new_id, traindevtest='test')
            new_id += 1
            make(back + v1, zeros, new_id, traindevtest='dev')
            make(back + v1, zeros, new_id, traindevtest='test') 
            new_id += 1

# 18,000 samples
def build_fourth_dataset():
    # fix random seed for reproducibility
    np.random.seed(10)

    # make directories
    base = os.path.join('data', 'fourth')
    if not os.path.exists(base):
        os.mkdir(base)
    for folder in ['train', 'dev', 'test']:
        path = os.path.join(base, folder)
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(os.path.join(path, 'input'))
            os.mkdir(os.path.join(path, 'label'))

    transforms = [RandomCrop(size=512), Downsample(scale=(1 / 2.0)), RotateFlip(), Normalize2(maxval=10**5)]

    def apply_transforms(sample_dict):
        for t in transforms:
            sample_dict = t.transform(sample_dict)
        return sample_dict

    fast = SherlockDataFast()

    new_id = 0

    def make(inp, lab, new_id, traindevtest='train'):
        sample_dict = {
            'input': inp,
            'label': lab
        }
        sample_dict = apply_transforms(sample_dict)
        input_path = os.path.join(base, traindevtest, 'input', str(new_id))
        np.save(input_path, sample_dict['input'])
        label_path = os.path.join(base, traindevtest, 'label', str(new_id))
        np.save(label_path, sample_dict['label'])
    
    for background_id in range(90):
        back = fast.get_background(background_id)
        for mask_id in range(10):
            v1 = fast.get_mask(background_id, mask_id, variable=1)
            v2 = fast.get_mask(background_id, mask_id, variable=2)
            v3 = fast.get_mask(background_id, mask_id, variable=3)
            zeros = np.zeros(v1.shape)
            for _ in range(10): # different crops, flips, rotations
                make(back + v2, v3, new_id, traindevtest='train') # variable
                new_id += 1
                make(back + v1, zeros, new_id, traindevtest='train') # nonvariable
                new_id += 1

    new_id = 0
    for background_id in range(90, 100):
        back = fast.get_background(background_id)
        for mask_id in range(10):
            v1 = fast.get_mask(background_id, mask_id, variable=1)
            v2 = fast.get_mask(background_id, mask_id, variable=2)
            v3 = fast.get_mask(background_id, mask_id, variable=3)
            zeros = np.zeros(v1.shape)

            make(back + v2, v3, new_id, traindevtest='dev')
            make(back + v2, v3, new_id, traindevtest='test')
            new_id += 1
            make(back + v1, zeros, new_id, traindevtest='dev')
            make(back + v1, zeros, new_id, traindevtest='test') 
            new_id += 1


class SherlockDataFast:
    EXP3 = '/scratch/users/dthomas5/exp3'
    SHAPE = (4072, 4000)

    def get_background(self, background_id):
        file = os.path.join(SherlockData.EXP3,
            'base',
            'output',
            'lsst_e_{}_f2_R22_S11_E000.fits'.format(background_id))
        back = fits.open(file)[0].data
        return back

    def get_mask(self, background_id, mask_id, variable=1):
        file = os.path.join(SherlockData.EXP3, 
            'variable{}'.format(variable),
            'mask{}'.format(mask_id),
            'output',
            'lsst_e_{}_f2_R22_S11_E000.fits'.format(background_id))
        mask = fits.open(file)[0].data
        return mask


# First epoch of a dataset with no downsampling
def build_third_dataset():
    # fix random seed for reproducibility
    np.random.seed(10)

    # make directories
    base = os.path.join('data', 'third')
    if not os.path.exists(base):
        os.mkdir(base)
    for folder in ['train', 'dev', 'test']:
        path = os.path.join(base, folder)
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(os.path.join(path, 'input'))
            os.mkdir(os.path.join(path, 'label'))

    transforms = [RandomCrop(size=512), RotateFlip(), Normalize2(maxval=10**5)]

    def apply_transforms(sample_dict):
        for t in transforms:
            sample_dict = t.transform(sample_dict)
        return sample_dict

    sherlock_data = SherlockData()

    new_id = 0

    def make(background_id, mask_id, new_id, variable=True, traindevtest='train'):
        if variable:
            sample_dict = {
                'input': sherlock_data.get_variable(background_id, mask_id),
                'label': sherlock_data.get_variable_label(background_id, mask_id)
            }
        else:
            sample_dict = {
                'input': sherlock_data.get_nonvariable(background_id, mask_id),
                'label': sherlock_data.get_nonvariable_label(background_id, mask_id)
            }
        sample_dict = apply_transforms(sample_dict)
        input_path = os.path.join(base, traindevtest, 'input', str(new_id))
        np.save(input_path, sample_dict['input'])
        label_path = os.path.join(base, traindevtest, 'label', str(new_id))
        np.save(label_path, sample_dict['label'])
    
    for _ in range(1): # different crops, flips, rotations
        for background_id in range(90):
            for mask_id in range(10):
                make(background_id, mask_id, new_id, variable=True, traindevtest='train')
                new_id += 1
                make(background_id, mask_id, new_id, variable=False, traindevtest='train')
                new_id += 1
            print(background_id)
    
    # dev 100  and test 100
    new_id = 0
    for background_id in range(90, 100):
        for mask_id in range(10):
            make(background_id, mask_id, new_id, variable=True, traindevtest='dev')
            make(background_id, mask_id, new_id, variable=True, traindevtest='test')
            new_id += 1
            make(background_id, mask_id, new_id, variable=False, traindevtest='dev')
            make(background_id, mask_id, new_id, variable=False, traindevtest='test')
            new_id += 1

def build_second_dataset():
    # fix random seed for reproducibility
    np.random.seed(10)

    # make directories
    base = os.path.join('data', 'second')
    if not os.path.exists(base):
        os.mkdir(base)
    for folder in ['train', 'dev', 'test']:
        path = os.path.join(base, folder)
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(os.path.join(path, 'input'))
            os.mkdir(os.path.join(path, 'label'))

    transforms = [RandomCrop(size=512), RotateFlip(), Normalize2(maxval=10**5), Downsample(scale=(1 / 4.0))]

    def apply_transforms(sample_dict):
        for t in transforms:
            sample_dict = t.transform(sample_dict)
        return sample_dict

    sherlock_data = SherlockData()

    new_id = 0

    def make(background_id, mask_id, new_id, variable=True, traindevtest='train'):
        if variable:
            sample_dict = {
                'input': sherlock_data.get_variable(background_id, mask_id),
                'label': sherlock_data.get_variable_label(background_id, mask_id)
            }
        else:
            sample_dict = {
                'input': sherlock_data.get_nonvariable(background_id, mask_id),
                'label': sherlock_data.get_nonvariable_label(background_id, mask_id)
            }
        sample_dict = apply_transforms(sample_dict)
        input_path = os.path.join(base, traindevtest, 'input', str(new_id))
        np.save(input_path, sample_dict['input'])
        label_path = os.path.join(base, traindevtest, 'label', str(new_id))
        np.save(label_path, sample_dict['label'])
    
    # train 10k
    for _ in range(10): # different crops, flips, rotations
        for background_id in range(90):
            for mask_id in range(10):
                make(background_id, mask_id, new_id, variable=True, traindevtest='train')
                new_id += 1
                make(background_id, mask_id, new_id, variable=False, traindevtest='train')
                new_id += 1
    
    # dev 100  and test 100
    new_id = 0
    for background_id in range(90, 100):
        for mask_id in range(10):
            make(background_id, mask_id, new_id, variable=True, traindevtest='dev')
            make(background_id, mask_id, new_id, variable=True, traindevtest='test')
            new_id += 1
            make(background_id, mask_id, new_id, variable=False, traindevtest='dev')
            make(background_id, mask_id, new_id, variable=False, traindevtest='test')
            new_id += 1

def build_initial_dataset():
    # fix random seed for reproducibility
    np.random.seed(10)

    # make directories
    base = os.path.join('data', 'initial')
    if not os.path.exists(base):
        os.mkdir(base)
    for folder in ['train', 'dev', 'test']:
        path = os.path.join(base, folder)
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(os.path.join(path, 'input'))
            os.mkdir(os.path.join(path, 'label'))

    transforms = [RandomCrop(size=512), RotateFlip(), Normalize(), Downsample(scale=(1 / 8.0))]

    def apply_transforms(sample_dict):
        for t in transforms:
            sample_dict = t.transform(sample_dict)
        return sample_dict

    sherlock_data = SherlockData()

    new_id = 0

    def make(background_id, mask_id, new_id, variable=True, traindevtest='train'):
        if variable:
            sample_dict = {
                'input': sherlock_data.get_variable(background_id, mask_id),
                'label': sherlock_data.get_variable_label(background_id, mask_id)
            }
        else:
            sample_dict = {
                'input': sherlock_data.get_nonvariable(background_id, mask_id),
                'label': sherlock_data.get_nonvariable_label(background_id, mask_id)
            }
        sample_dict = apply_transforms(sample_dict)
        input_path = os.path.join(base, traindevtest, 'input', str(new_id))
        np.save(input_path, sample_dict['input'])
        label_path = os.path.join(base, traindevtest, 'label', str(new_id))
        np.save(label_path, sample_dict['label'])
    
    # train 10k
    # for _ in range(10): # different crops, flips, rotations
    #     for background_id in range(90):
    #         for mask_id in range(10):
    #             make(background_id, mask_id, new_id, variable=True, traindevtest='train')
    #             new_id += 1
    #             make(background_id, mask_id, new_id, variable=False, traindevtest='train')
    #             new_id += 1
    
    # dev 100  and test 100
    new_id = 0
    for background_id in range(90, 100):
        for mask_id in range(10):
            make(background_id, mask_id, new_id, variable=True, traindevtest='dev')
            make(background_id, mask_id, new_id, variable=True, traindevtest='test')
            new_id += 1
            make(background_id, mask_id, new_id, variable=False, traindevtest='dev')
            make(background_id, mask_id, new_id, variable=False, traindevtest='test')
            new_id += 1


class SherlockData:
    EXP3 = '/scratch/users/dthomas5/exp3'
    SHAPE = (4072, 4000)

    def _get_background(self, background_id):
        file = os.path.join(SherlockData.EXP3,
            'base',
            'output',
            'lsst_e_{}_f2_R22_S11_E000.fits'.format(background_id))
        back = fits.open(file)[0].data
        return back

    def _get_mask(self, background_id, mask_id, variable=1):
        file = os.path.join(SherlockData.EXP3, 
            'variable{}'.format(variable),
            'mask{}'.format(mask_id),
            'output',
            'lsst_e_{}_f2_R22_S11_E000.fits'.format(background_id))
        mask = fits.open(file)[0].data
        return mask

    def get_variable(self, background_id, mask_id):
        back = self._get_background(background_id)
        mask = self._get_mask(background_id, mask_id, variable=2)
        return back + mask

    def get_variable_label(self, background_id, mask_id):
        mask = self._get_mask(background_id, mask_id, variable=3)
        return mask

    def get_nonvariable(self, background_id, mask_id):
        back = self._get_background(background_id)
        mask = self._get_mask(background_id, mask_id, variable=1)
        return back + mask

    def get_nonvariable_label(self, background_id, mask_id):
        return np.zeros(SherlockData.SHAPE)

class Transform:
    def transform(self, sample_dict):
        raise NotImplementedError()

class CopyCrop(Transform):
    def __init__(self, crop_to_copy, size=512):
        self.size = size
        self.crop_to_copy = crop_to_copy

    def transform(self, sample_dict):
        inp, lab = sample_dict['input'], sample_dict['label']
        x,y = self.crop_to_copy.crop
        inp = inp[(x-self.size // 2):(x+self.size // 2),
                  (y-self.size // 2):(y+self.size // 2)]
        lab = lab[(x-self.size // 2):(x+self.size // 2),
                  (y-self.size // 2):(y+self.size // 2)]
        self.crop = (x,y)
        output = {'input': inp, 'label': lab}
        return output


class SmartCrop(Transform):
    def __init__(self, size=512):
        self.size = size
        self.crop = None

    def transform(self, sample_dict):
        inp, lab = sample_dict['input'], sample_dict['label']
        bigx, bigy = inp.shape

        pixx, pixy = np.nonzero(lab > 5)
        while True:
            ind = np.random.randint(0, len(pixx))
            x = pixx[ind]
            y = pixy[ind]

            if self.size // 2 < x < bigx - self.size // 2 and \
                self.size // 2 < y < bigy - self.size // 2:
                inp = inp[(x-self.size // 2):(x+self.size // 2),
                          (y-self.size // 2):(y+self.size // 2)]
                lab = lab[(x-self.size // 2):(x+self.size // 2),
                          (y-self.size // 2):(y+self.size // 2)]
                print('labsum', lab.sum())
                self.crop = (x,y)
                output = {'input': inp, 'label': lab}
                return output

class RandomCrop(Transform):
    def __init__(self, size=512):
        self.size = size

    def transform(self, sample_dict):
        inp, lab = sample_dict['input'], sample_dict['label']
        bigx, bigy = inp.shape
        x = np.random.randint(0, bigx - self.size)
        y = np.random.randint(0, bigy - self.size)
        inp = inp[x:(x + self.size), y: (y + self.size)]
        lab = lab[x:(x + self.size), y: (y + self.size)]
        output = {'input': inp, 'label': lab}
        return output

class Normalize(Transform):
    def transform(self, sample_dict):
        inp, lab = sample_dict['input'], sample_dict['label']
        inp, lab = np.log(inp + 1), np.log(lab + 1)
        output = {'input': inp, 'label': lab}
        return output

class Normalize2(Transform):
    def __init__(self, maxval=10**5):
        self.maxval = maxval

    def transform(self, sample_dict):
        inp, lab = sample_dict['input'], sample_dict['label']
        maxlog = np.log(self.maxval)
        inp = np.log(inp + 1) / maxlog
        lab = np.log(lab + 1) / maxlog
        output = {'input': inp, 'label': lab}
        return output

class RotateFlip(Transform):
    def transform(self, sample_dict):
        inp, lab = sample_dict['input'], sample_dict['label']
        if np.random.uniform() > 0.5:
            inp = np.fliplr(inp)
            lab = np.fliplr(lab)
        rot = np.random.randint(4)
        inp = np.rot90(inp, k=rot)
        lab = np.rot90(lab, k=rot)
        output = {'input': inp, 'label': lab}
        return output

# from scipy.ndimage.interpolation import rotate # THIS IS SUPER SLOW

# class RotateFlip(Transform):
#     def __init__(self, angles=[0, 90, 180, 270]):
#         self.angles = angles

#     def transform(self, sample_dict):
#         inp, lab = sample_dict['input'], sample_dict['label']
#         if np.random.uniform() > 0.5:
#             inp = np.fliplr(inp)
#             lab = np.fliplr(lab)
#         rot = np.random.randint(4)
#         inp = rotate(inp, self.angles[rot], order=1)
#         lab = rotate(lab, self.angles[rot], order=1)
#         output = {'input': inp, 'label': lab}
#         return output

class Downsample(Transform):
    def __init__(self, scale=(1 / 8.0)):
        self.scale = scale

    def transform(self, sample_dict):
        inp, lab = sample_dict['input'], sample_dict['label']
        inp = rescale(inp.astype('double'), self.scale, order=1, preserve_range=False, clip=False)
        lab = rescale(lab.astype('double'), self.scale, order=1, preserve_range=False, clip=False)
        output = {'input': inp.astype('float32'), 'label': lab.astype('float32')}
        return output        

if __name__ == '__main__':
    build_durtest_dataset()
