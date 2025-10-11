import os.path
import torch
import scipy.io as io
from scipy import ndimage
import numpy as np
import torch.utils.data as Data
import glob
import os

root = '.../data/'
train_root = root+'train/'
test_root = root+'test/'
PiB_root = root+'test/PiBVis/'

def pet_loader(path):
    mat = io.loadmat(path)

    name = os.path.basename(path)
    clean = mat['clean']
    noisy = mat['noisy']
    mr = mat['mr']

    return clean, noisy, mr, name

def pib_loader(path):
    mat = io.loadmat(path)

    name = os.path.basename(path)
    clean = mat['clean']
    noisy = mat['noisy']
    mr = mat['mr']
    seg = mat['seg']
    isub = mat['isub']

    return clean, noisy, mr, seg, isub, name

class PETData(Data.Dataset):
    def __init__(self, root, transform=None, loader=pet_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.imgs = glob.glob(root + "*.mat")

    def __getitem__(self, index):
        img_path = self.imgs[index]
        clean, noisy, mr, name = self.loader(img_path)

        clean = torch.FloatTensor(torch.from_numpy(clean).float())
        noisy = torch.FloatTensor(torch.from_numpy(noisy).float())
        mr = torch.FloatTensor(torch.from_numpy(mr).float())

        clean = torch.unsqueeze(clean, 0)
        noisy = torch.unsqueeze(noisy, 0)
        mr = torch.unsqueeze(mr, 0)

        return {'clean': clean, 'mr': mr, 'noisy': noisy}, name

    def __len__(self):
        return len(self.imgs)

class PiBData(Data.Dataset):
    def __init__(self, root, transform=None, loader=pib_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.imgs = glob.glob(root + "*.mat")

    def __getitem__(self, index):
        img_path = self.imgs[index]
        clean, noisy, mr, seg, isub, name = self.loader(img_path)

        clean = torch.FloatTensor(torch.from_numpy(clean).float())
        noisy = torch.FloatTensor(torch.from_numpy(noisy).float())
        mr = torch.FloatTensor(torch.from_numpy(mr).float())
        seg = torch.FloatTensor(torch.from_numpy(seg).float())

        clean = torch.unsqueeze(clean, 0)
        noisy = torch.unsqueeze(noisy, 0)
        mr = torch.unsqueeze(mr, 0)
        seg = torch.unsqueeze(seg, 0)

        isub = isub[0]

        return {'clean': clean, 'mr': mr, 'noisy': noisy, 'seg': seg}, isub, name

    def __len__(self):
        return len(self.imgs)

def get_train_loaders(batch_size):
    train_data = Data.DataLoader(
        PETData(root=train_root + '/', transform=None),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    return train_data

def get_test_loader(scanner, tracer, batch_size):
    test_data = Data.DataLoader(
        PETData(root=test_root + scanner + '/' + tracer + '/', transform=None),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True)

    return test_data

def get_pib_loader(batch_size=1):
    test_data = Data.DataLoader(
        PiBData(root=PiB_root, transform=None),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True)

    return test_data

