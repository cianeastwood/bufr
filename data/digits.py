import torch
import os
import sys
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from lib.data_utils import train_test_split_subset
from data.mnistm import MNISTM, MNISTM_Idx
from data.mnist_c import MnistC, MnistC_Idx


def get_mnist_dataloaders(data_root, batch_size, shuffle, augment, n_workers=4, pin_mem=True, split_seed=None):
    """
    Mnist is 1x28x28. Train set is size 60000. Test set is size 10000.
    """
    if augment:
        transform = transforms.Compose([transforms.RandomRotation(10),
                                        transforms.Lambda(lambda x: x.convert("RGB")),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([transforms.Lambda(lambda x: x.convert("RGB")),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # No RGB
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.5,), (0.5,))])

    mnist_dataset = MNIST(data_root, train=True, transform=transform, download=True)
    tr_ds, val_ds = train_test_split_subset(mnist_dataset, seed=split_seed)
    tst_ds = MNIST(data_root, train=False, transform=transform, download=True)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)

    return tr_loader, val_loader, tst_loader


def get_mnistm_dataloaders(data_root, batch_size, shuffle, augment, n_workers=4, pin_mem=True, split_seed=None):
    """
    Mnist-m is 3x28x28. Train set is size 60000. Test set is size 10000. Same as mnist.
    """
    if augment:
        transform = transforms.Compose([transforms.RandomRotation(10),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    mnistm_dataset = MNISTM(os.path.join(data_root, "MNISTM"), mnist_root=data_root, train=True, transform=transform,
                            download=True)
    tr_ds, val_ds = train_test_split_subset(mnistm_dataset, seed=split_seed)
    tst_ds = MNISTM(os.path.join(data_root, "MNISTM"), mnist_root=data_root, train=False, transform=transform,
                    download=True)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)

    return tr_loader, val_loader, tst_loader


def get_mnistm_idx_dataloaders(data_root, batch_size, shuffle, augment, n_workers=4, pin_mem=True, split_seed=None):
    """
    Mnist-m is 3x28x28. Train set is size 60000. Test set is size 10000. Same as mnist.
    """
    if augment:
        transform = transforms.Compose([transforms.RandomRotation(10),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    mnistm_dataset = MNISTM_Idx(os.path.join(data_root, "MNISTM"), mnist_root=data_root, train=True,
                                transform=transform, download=True)
    # don't split tr_ds otherwise pseudo labelling breaks
    tr_ds, val_ds = train_test_split_subset(mnistm_dataset, train_split=1.0, seed=split_seed)
    tst_ds = MNISTM_Idx(os.path.join(data_root, "MNISTM"), mnist_root=data_root, train=False, transform=transform,
                    download=True)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=False)
    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=False)

    return tr_loader, val_loader, tst_loader


def get_mnist_c_dataloaders(data_root, batch_size, shuffle, augment, corruption="stripe", n_workers=4, pin_mem=True,
                            split_seed=None):
    """
    Mnist with corruptions is 1x28x28 with 15 different corruptions. Train set is size 60000. Test set is size 10000.
    """
    if augment:
        transform = transforms.Compose([transforms.RandomRotation(10),
                                        transforms.Lambda(lambda x: x.convert("RGB")),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([transforms.Lambda(lambda x: x.convert("RGB")),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    mnist_dataset = MnistC(os.path.join(data_root, "mnist_c"), corruption=corruption, train=True, transform=transform)
    tr_ds, val_ds = train_test_split_subset(mnist_dataset, seed=split_seed)
    tst_ds = MnistC(os.path.join(data_root, "mnist_c"), corruption=corruption, train=False, transform=transform)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)

    return tr_loader, val_loader, tst_loader


def get_mnist_c_idx_dataloaders(data_root, batch_size, shuffle, augment, corruption="stripe", n_workers=4, pin_mem=True,
                                split_seed=None):
    """
    Mnist with corruptions is 1x28x28 with 15 different corruptions. Train set is size 60000. Test set is size 10000.
    """
    if augment:
        transform = transforms.Compose([transforms.RandomRotation(10),
                                        transforms.Lambda(lambda x: x.convert("RGB")),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([transforms.Lambda(lambda x: x.convert("RGB")),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    mnist_dataset = MnistC_Idx(os.path.join(data_root, "mnist_c"), corruption=corruption, train=True,
                               transform=transform)
    tr_ds, val_ds = train_test_split_subset(mnist_dataset, train_split=1.0, seed=split_seed)
    tst_ds = MnistC_Idx(os.path.join(data_root, "mnist_c"), corruption=corruption, train=False, transform=transform)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=False)
    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=False)

    return tr_loader, val_loader, tst_loader