from __future__ import division, print_function, absolute_import
import os
import csv
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder, MNIST
from lib.data_transforms import normalize, normalize_0_1
from data.emnist import StaticEMNIST, DynamicEMNIST, StaticEMNIST_Idx
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.datasets.wilds_dataset import WILDSSubset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
CAMELYON_MEAN = (0.485, 0.456, 0.406)
CAMELYON_STD = (0.229, 0.224, 0.225)


def get_dataset_labels(dataset):
    if isinstance(dataset, ImageFolder) or isinstance(dataset, MNIST):
        return np.array(dataset.targets)            # avoid type(labels[0]) == tensor
    elif isinstance(dataset, Subset):
        # Subset holds the actual dataset as an attribute, and the subset indices as another.
        if isinstance(dataset.dataset, ImageFolder):
            orig_targets = dataset.dataset.targets
        else:
            raise Exception("Unable to retrieve labels indirectly -- please pass in a tensor of labels (directly).")
        subset_indices = np.array(dataset.indices)
        return list(orig_targets[subset_indices])
    else:
        raise Exception("You should pass the tensor of labels to the constructor as second argument")


def get_dataset_label(dataset, idx):
    if isinstance(dataset, ImageFolder) or isinstance(dataset, MNIST):
        return dataset.targets[idx]
    elif isinstance(dataset, Subset):
        # Subset holds the actual dataset as an attribute, and the subset indices as another.
        return get_dataset_label(dataset.dataset, dataset.indices[idx])
    else:
        raise Exception("Unable to retrieve labels indirectly -- please pass in a tensor of labels (directly).")


def per_class_subset(dataset, n_samples_per_class=5):
    # Extract basic dataset info
    labels = get_dataset_labels(dataset)
    classes, original_n_examples_per_class = np.unique(labels, return_counts=True)
    num_classes = len(classes)

    # Group all the samples indices for each class using a dict
    label_to_sample_inds = {}
    for idx in range(0, len(dataset)):
        label = labels[idx]
        if label not in label_to_sample_inds:
            label_to_sample_inds[label] = list()
        label_to_sample_inds[label].append(idx)

    # Create list of n_samples_per_class
    if isinstance(n_samples_per_class, int):  # balanced dataset with n examples per class
        per_class_n = [n_samples_per_class] * num_classes
    else:                   # provide a sequence of numbers indicating desired n examples per class
        per_class_n = list(n_samples_per_class)
        if len(per_class_n) != num_classes:
            raise ValueError("Length of sequence provided ({0}) does not match the number of classes in the dataset"
                             " ({1})".format(len(per_class_n), num_classes))

    # Replace *all* samples indices for each class with a random sample of size n
    for label, c_n in zip(label_to_sample_inds, per_class_n):
        # Oversample the classes with fewer elements than the number desired c_n
        existing_c_n = len(label_to_sample_inds[label])
        if existing_c_n < c_n:
            try:
                # Choose m additional samples from n existing, without replacement (assumes m <= n)
                n_additional_samples = c_n - existing_c_n
                additional_samples = np.random.choice(label_to_sample_inds[label],
                                                      size=n_additional_samples, replace=False)
                label_to_sample_inds[label].extend(additional_samples)
            except ValueError():
                print("Warning: oversampling with replacement.")
                # Sample m additional samples with replacement (no assumption on m, n)
                while len(label_to_sample_inds[label]) < c_n:
                    label_to_sample_inds[label].append(np.random.choice(label_to_sample_inds[label]))

        # Sample the desired number of examples from each class
        label_to_sample_inds[label] = np.random.choice(label_to_sample_inds[label], c_n, replace=False)

    # Extract to sampled indices into a single list
    inds_to_keep = []
    for label, inds in label_to_sample_inds.items():
        inds_to_keep.extend(inds)
    inds_to_keep.sort()

    # Create a new (sub)dataset with these indices
    new_ds = Subset(dataset, inds_to_keep)

    # Print some stats
    print("Examples-per-class:")
    print("\tOld", list(original_n_examples_per_class))
    print("\tNew", per_class_n)

    return new_ds


def get_dynamic_emnist_dataloaders(dataset_path, keep_classes, batch_size, shuffle, n_workers, pin_mem):
    tr_ds = DynamicEMNIST(dataset_path, norm=True, color=True, which_set='train',
                          keep_classes=keep_classes)
    val_ds = DynamicEMNIST(dataset_path, norm=True, color=True, which_set='valid',
                           keep_classes=keep_classes)
    tst_ds = DynamicEMNIST(dataset_path, norm=True, color=True, which_set='test',
                           keep_classes=keep_classes)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)

    return tr_loader, val_loader, tst_loader


def train_test_split_subset(ds, train_split=0.9, shuffle=True, seed=None):
    ds_size = len(ds)
    if shuffle:
        if seed is not None:
            g_cpu = torch.Generator()
            g_cpu.manual_seed(seed)
        else:
            g_cpu = None
        ds_inds = torch.randperm(ds_size, generator=g_cpu)
    else:
        ds_inds = torch.arange(ds_size)

    if train_split < 1:
        tr_size = int(ds_size * train_split)
        tr_inds, val_inds = ds_inds[:tr_size], ds_inds[tr_size:]
    else:
        tr_inds = ds_inds
        val_size = int(0.1 * ds_size)  # still gives 10% test data but now overlaps with the training set
        val_inds = ds_inds[-val_size:]

    tr_ds = Subset(ds, ds_inds[tr_inds])
    val_ds = Subset(ds, ds_inds[val_inds])

    return tr_ds, val_ds


def train_val_test_split_subset(ds, train_split=0.8, shuffle=True):
    ds_size = len(ds)
    if shuffle:
        ds_inds = torch.randperm(ds_size)
    else:
        ds_inds = torch.arange(ds_size)

    assert train_split <= 0.8
    tr_size = int(ds_size * train_split)
    val_split = round((1. - train_split) / 2., 1)
    val_size = int(ds_size * val_split)
    tr_inds, val_inds, tst_inds = ds_inds[:tr_size], ds_inds[tr_size:tr_size + val_size], ds_inds[tr_size + val_size:]

    tr_ds = Subset(ds, ds_inds[tr_inds])
    val_ds = Subset(ds, ds_inds[val_inds])
    tst_ds = Subset(ds, ds_inds[tst_inds])

    return tr_ds, val_ds, tst_ds


def train_val_test_split_wilds_subset(ds, train_split=0.8, shuffle=True):
    ds_size = len(ds)
    if shuffle:
        ds_inds = torch.randperm(ds_size)
    else:
        ds_inds = torch.arange(ds_size)

    assert train_split <= 0.8
    tr_size = int(ds_size * train_split)
    val_split = round((1. - train_split) / 2., 1)
    val_size = int(ds_size * val_split)
    tr_inds, val_inds, tst_inds = ds_inds[:tr_size], ds_inds[tr_size:tr_size + val_size], ds_inds[tr_size + val_size:]

    tr_ds = WILDSSubset(ds, ds_inds[tr_inds], transform=None)
    val_ds = WILDSSubset(ds, ds_inds[val_inds], transform=None)
    tst_ds = WILDSSubset(ds, ds_inds[tst_inds], transform=None)

    return tr_ds, val_ds, tst_ds


def per_hospital_wilds_dataloader(dataset_path, hospital_idx, batch_size=32, n_workers=4, pin_mem=True):
    dataset = get_dataset(dataset='camelyon17', download=True, root_dir=dataset_path + "CAMELYON17", version="1.0")
    # print(dataset.metadata_fields)  # ['hospital', 'slide', 'y']

    with open(dataset_path + "CAMELYON17/camelyon17_v1.0/metadata.csv", newline='') as csvfile:
        metadata_reader = csv.reader(csvfile)
        header_row = True
        # https://stackoverflow.com/questions/8713620/appending-items-to-a-list-of-lists-in-python
        center_indices = [[] for _ in range(5)]
        for row in metadata_reader:
            if header_row:
                center_csv_idx = row.index('center')
                datapoint_csv_idx = 0
                header_row = False
            else:
                center_indices[int(row[center_csv_idx])].append(int(row[datapoint_csv_idx]))

    # Get the training set
    wilds_transforms = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(CAMELYON_MEAN, CAMELYON_STD)
    ])

    data = WILDSSubset(dataset, center_indices[hospital_idx], transform=wilds_transforms)

    # Shuffle
    loader = get_train_loader('standard', data, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_mem)

    return loader


def per_hospital_wilds_dataloader_split(dataset_path, hospital_idx, batch_size=32, n_workers=4, pin_mem=True):
    dataset = get_dataset(dataset='camelyon17', download=True, root_dir=dataset_path + "CAMELYON17", version="1.0")
    # print(dataset.metadata_fields)  # ['hospital', 'slide', 'y']

    with open(dataset_path + "CAMELYON17/camelyon17_v1.0/metadata.csv", newline='') as csvfile:
        metadata_reader = csv.reader(csvfile)
        header_row = True
        # https://stackoverflow.com/questions/8713620/appending-items-to-a-list-of-lists-in-python
        center_indices = [[] for _ in range(5)]
        for row in metadata_reader:
            if header_row:
                center_csv_idx = row.index('center')
                datapoint_csv_idx = 0
                header_row = False
            else:
                center_indices[int(row[center_csv_idx])].append(int(row[datapoint_csv_idx]))

    # Get the training set
    wilds_transforms = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(CAMELYON_MEAN, CAMELYON_STD)
    ])

    data = WILDSSubset(dataset, center_indices[hospital_idx], transform=wilds_transforms)

    tr_data, val_data, tst_data = train_val_test_split_wilds_subset(data)

    tr_dl = get_train_loader('standard', tr_data, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_mem)

    val_dl = get_eval_loader('standard', val_data, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_mem)

    tst_dl = get_eval_loader('standard', tst_data, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_mem)

    return tr_dl, val_dl, tst_dl


def per_hospital_wilds_dataloader_fewshot(dataset_path, hospital_idx, n_samples_per_class, batch_size=32,
                                          n_workers=4, pin_mem=True):
    dataset = get_dataset(dataset='camelyon17', download=True, root_dir=dataset_path + "CAMELYON17", version="1.0")
    # print(dataset.metadata_fields)  # ['hospital', 'slide', 'y']

    with open(dataset_path + "CAMELYON17/camelyon17_v1.0/metadata.csv", newline='') as csvfile:
        metadata_reader = csv.reader(csvfile)
        header_row = True
        # https://stackoverflow.com/questions/8713620/appending-items-to-a-list-of-lists-in-python
        center_indices = [[] for _ in range(5)]
        tumor_labels = [[] for _ in range(5)]
        for row in metadata_reader:
            if header_row:
                center_csv_idx = row.index('center')
                tumor_csv_idx = row.index('tumor')
                datapoint_csv_idx = 0
                header_row = False
            else:
                center_indices[int(row[center_csv_idx])].append(int(row[datapoint_csv_idx]))
                tumor_labels[int(row[center_csv_idx])].append(int(row[tumor_csv_idx]))

    # Get the training set
    wilds_transforms = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(CAMELYON_MEAN, CAMELYON_STD)
    ])

    # We use a piece of hardcoding that all 1 labels are in first half of the metadata file
    hospital_indices = center_indices[hospital_idx]
    assert len(hospital_indices) % 2 == 0
    positive_indices = hospital_indices[:len(hospital_indices) // 2]
    negative_indices = hospital_indices[len(hospital_indices) // 2:]
    positive_samples = list(np.random.choice(positive_indices, n_samples_per_class, replace=False))
    negative_samples = list(np.random.choice(negative_indices, n_samples_per_class, replace=False))
    all_samples = positive_samples + negative_samples

    # Check
    print("Number of samples per class: {}".format(n_samples_per_class))
    positive_count = 0
    negative_count = 0
    for sample in all_samples:
        if tumor_labels[hospital_idx][center_indices[hospital_idx].index(sample)] == 1:
            positive_count += 1
        elif tumor_labels[hospital_idx][center_indices[hospital_idx].index(sample)] == 0:
            negative_count += 1
        else:
            raise RuntimeError("This should never happen if tumor labels extracted correctly.")
    print("Generated number of samples for class 0: {}".format(negative_count))
    print("Generated number of samples for class 1: {}".format(positive_count))

    data = WILDSSubset(dataset, all_samples, transform=wilds_transforms)

    # Shuffle
    loader = get_train_loader('standard', data, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_mem)

    return loader


def get_cifar10_dataloaders(dataset_path, batch_size, shuffle, n_workers, pin_mem, train_split=0.9, normalize=True):
    # Setup
    assert 0.75 < train_split <= 1.
    ds_path = os.path.join(dataset_path, "CIFAR-10")

    # Create transforms
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    test_transforms = [transforms.ToTensor()]
    if normalize:
        norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        train_transforms.append(norm)
        test_transforms.append(norm)

    transform_train = transforms.Compose(train_transforms)
    transform_test = transforms.Compose(test_transforms)

    # Load datasets
    tr_ds = torchvision.datasets.CIFAR10(ds_path, train=True, transform=transform_train, download=True)
    tst_ds = torchvision.datasets.CIFAR10(ds_path, train=False, transform=transform_test, download=True)

    #  Train, valid set split via Subset()
    tr_ds, val_ds = train_test_split_subset(tr_ds, train_split, shuffle)

    # Create dataloaders
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=200, shuffle=False,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    tst_loader = DataLoader(tst_ds, batch_size=200, shuffle=False,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)

    return tr_loader, val_loader, tst_loader


def get_cifar100_dataloaders(dataset_path, batch_size, shuffle, n_workers, pin_mem, train_split=0.9, normalize=True):
    # Setup
    assert 0.75 < train_split <= 1.
    ds_path = os.path.join(dataset_path, "CIFAR-100")

    # Create transforms
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ]
    test_transforms = [transforms.ToTensor()]
    if normalize:
        norm = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        train_transforms.append(norm)
        test_transforms.append(norm)

    transform_train = transforms.Compose(train_transforms)
    transform_test = transforms.Compose(test_transforms)

    # Load datasets
    tr_ds = torchvision.datasets.CIFAR100(ds_path, train=True, transform=transform_train, download=True)
    tst_ds = torchvision.datasets.CIFAR100(ds_path, train=False, transform=transform_test, download=True)

    #  Train, valid set split via Subset()
    tr_ds, val_ds = train_test_split_subset(tr_ds, train_split, shuffle)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=200, shuffle=False,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    tst_loader = DataLoader(tst_ds, batch_size=200, shuffle=False,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)

    return tr_loader, val_loader, tst_loader


def get_cifar10c_dataloaders(dataset_path, corruption_name, severity, batch_size, shuffle, n_workers, pin_mem,
                             train_split=1., normalize=True):
    # Setup
    assert 1 <= severity <= 5
    assert 0.7 < train_split <= 1.
    ds_path = os.path.join(dataset_path, "CIFAR-10-C/")
    n_total_cifar = 10000

    # Load data
    all_xs = np.load(ds_path + corruption_name + '.npy')
    all_ys = torch.LongTensor(np.load(ds_path + 'labels.npy'))
    start_idx = (severity - 1) * n_total_cifar
    end_idx = severity * n_total_cifar
    xs = all_xs[start_idx:end_idx]
    ys = all_ys[start_idx:end_idx]

    # ToTensor()
    xs = xs.astype(np.float32) / 255.
    xs = np.transpose(xs, (0, 3, 1, 2))
    xs = torch.from_numpy(xs)

    # Normalize()
    if normalize:
        xs = normalize_cifar_torch(xs)

    # Create Tensor dataset
    ds = torch.utils.data.TensorDataset(xs, ys)

    # Train, valid set split via Subset()
    if train_split <= 0.8:               # Create proper train, val, test splits
        tr_ds, val_ds, tst_ds = train_val_test_split_subset(ds, train_split, shuffle)
    else:                                # Flawed but standard UDA setup
        tr_ds, val_ds = train_test_split_subset(ds, train_split, shuffle)
        tst_ds = None

    # Create dataloaders
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=200, shuffle=False,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    if tst_ds is not None:
        tst_loader = DataLoader(tst_ds, batch_size=200, shuffle=False,
                                num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
        return tr_loader, val_loader, tst_loader

    return tr_loader, val_loader


def get_cifar100c_dataloaders(dataset_path, corruption_name, severity, batch_size, shuffle, n_workers, pin_mem,
                              train_split=1., normalize=True):
    # Setup
    assert 1 <= severity <= 5
    assert 0.7 < train_split <= 1.
    ds_path = os.path.join(dataset_path, "CIFAR-100-C/")
    n_total_cifar = 10000

    # Load data
    all_xs = np.load(ds_path + corruption_name + '.npy')
    all_ys = torch.LongTensor(np.load(ds_path + 'labels.npy'))
    start_idx = (severity - 1) * n_total_cifar
    end_idx = severity * n_total_cifar
    xs = all_xs[start_idx:end_idx]
    ys = all_ys[start_idx:end_idx]

    # ToTensor()
    xs = xs.astype(np.float32) / 255.
    xs = np.transpose(xs, (0, 3, 1, 2))
    xs = torch.from_numpy(xs)

    # Normalize()
    if normalize:
        xs = normalize_cifar_torch(xs, False)

    # Create Tensor dataset
    ds = torch.utils.data.TensorDataset(xs, ys)

    # Train, valid, (maybe test) set split via Subset()
    if train_split <= 0.8:  # Create proper train, val, test splits
        tr_ds, val_ds, tst_ds = train_val_test_split_subset(ds, train_split, shuffle)
    else:  # Flawed but standard UDA setup
        tr_ds, val_ds = train_test_split_subset(ds, train_split, shuffle)
        tst_ds = None

    # Create dataloaders
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=200, shuffle=False,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    if tst_ds is not None:
        tst_loader = DataLoader(tst_ds, batch_size=200, shuffle=False,
                                num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
        return tr_loader, val_loader, tst_loader

    return tr_loader, val_loader


def normalize_cifar_torch(xs, is_cifar_10=True):
    if is_cifar_10:
        m = torch.tensor(CIFAR10_MEAN)
        std = torch.tensor(CIFAR10_STD)
    else:
        m = torch.tensor(CIFAR100_MEAN)
        std = torch.tensor(CIFAR100_STD)
    xs = xs.transpose(1, 3)             # [B,C,H,W] --> [B,W,H,C]
    xs = (xs - m) / std                 # [B,W,H,C] / [C]: Required shapes for PyTorch broadcasting
    xs = xs.transpose(1, 3)             # [B,W,H,C] --> [B,C,H,W]
    return xs


def get_static_emnist_dataloaders(dataset_path, keep_classes, batch_size, shuffle, n_workers, pin_mem):
    tr_ds = StaticEMNIST(dataset_path, keep_classes, which_set='train',
                         transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))
    val_ds = StaticEMNIST(dataset_path, keep_classes, which_set='valid',
                          transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))
    tst_ds = StaticEMNIST(dataset_path, keep_classes, which_set='test',
                          transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)

    return tr_loader, val_loader, tst_loader


def get_static_emnist_dataloaders_fewshot(dataset_path, n_samples_per_class, keep_classes, batch_size, shuffle,
                                          n_workers, pin_mem):
    tr_ds = StaticEMNIST(dataset_path, keep_classes, which_set='train',
                         transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))
    val_ds = StaticEMNIST(dataset_path, keep_classes, which_set='valid',
                          transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))
    tst_ds = StaticEMNIST(dataset_path, keep_classes, which_set='test',
                          transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))

    total_n_classes = 47  # hardcoded for EMNIST all classes
    print("\n ------------- {0} -------------".format("train"))
    print("Original len:", len(tr_ds))
    print("Original avg-n-shot-per-class:", len(tr_ds) / total_n_classes)
    tr_ds = per_class_subset(tr_ds, n_samples_per_class)
    print("New len", len(tr_ds))
    print("New avg-n-shot-per-class:", len(tr_ds) / total_n_classes)

    print("\n ------------- {0} -------------".format("valid"))
    print("Original len:", len(val_ds))
    print("Original avg-n-shot-per-class:", len(val_ds) / total_n_classes)
    val_ds = per_class_subset(val_ds, n_samples_per_class)
    print("New len", len(val_ds))
    print("New avg-n-shot-per-class:", len(val_ds) / total_n_classes)

    print("\n ------------- {0} -------------".format("test"))
    print("Original len:", len(tst_ds))
    print("Original avg-n-shot-per-class:", len(tst_ds) / total_n_classes)
    tst_ds = per_class_subset(tst_ds, n_samples_per_class)
    print("New len", len(tst_ds))
    print("New avg-n-shot-per-class:", len(tst_ds) / total_n_classes)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)

    return tr_loader, val_loader, tst_loader


def get_static_emnist_dataloaders_oracle(dataset_path, keep_classes, batch_size, shuffle, n_workers, pin_mem):
    tst_ds = StaticEMNIST(dataset_path, keep_classes, which_set='test',
                          transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))

    tr_ds, val_ds, tst_ds = train_val_test_split_subset(tst_ds)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)
    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=True)

    return tr_loader, val_loader, tst_loader


def get_static_emnist_idx_dataloaders(dataset_path, keep_classes, batch_size, shuffle, n_workers, pin_mem):
    tr_ds = StaticEMNIST_Idx(dataset_path, keep_classes, which_set='train',
                         transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))
    val_ds = StaticEMNIST_Idx(dataset_path, keep_classes, which_set='valid',
                          transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))
    tst_ds = StaticEMNIST_Idx(dataset_path, keep_classes, which_set='test',
                          transform=torchvision.transforms.Compose([ToTensor(), normalize_0_1]))

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=n_workers, pin_memory=pin_mem, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=False)
    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=n_workers, pin_memory=pin_mem, drop_last=False)

    return tr_loader, val_loader, tst_loader
