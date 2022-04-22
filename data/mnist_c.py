from __future__ import print_function
import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class MnistC(data.Dataset):
    """Mnist with different background images"""

    training_images = "train_images.npy"
    training_labels = "train_labels.npy"
    testing_images = "test_images.npy"
    testing_labels = "test_labels.npy"

    def __init__(self, root, corruption, train=True, transform=None, target_transform=None):
        super(MnistC, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        if corruption not in ["brightness", "canny_edges", "dotted_line", "fog", "glass_blur", "identity",
                              "impulse_noise", "motion_blur", "rotate", "scale", "shear", "shot_noise", "spatter",
                              "stripe", "translate", "zigzag"]:
            raise ValueError("Invalid corruption name chosen")
        else:
            self.corruption_folder = corruption

        if not self._check_exists():
            raise RuntimeError("Mnist corrupted ({}) dataset not found".format(self.corruption_folder))

        if self.train:
            self.train_data = torch.tensor(np.load(os.path.join(self.root, self.corruption_folder,
                                                                self.training_images)))
            self.train_labels = torch.tensor(np.load(os.path.join(self.root, self.corruption_folder,
                                                                  self.training_labels)))
        else:
            self.test_data = torch.tensor(np.load(os.path.join(self.root, self.corruption_folder,
                                                               self.testing_images)))
            self.test_labels = torch.tensor(np.load(os.path.join(self.root, self.corruption_folder,
                                                                 self.testing_labels)))

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.squeeze().numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.corruption_folder, self.training_images)) and \
               os.path.exists(os.path.join(self.root, self.corruption_folder, self.training_labels)) and \
               os.path.exists(os.path.join(self.root, self.corruption_folder, self.testing_images)) and \
               os.path.exists(os.path.join(self.root, self.corruption_folder, self.testing_labels))


class MnistC_Idx(MnistC):
    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.squeeze().numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index