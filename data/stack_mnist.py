# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import numpy as np
from PIL import Image
import torchvision.datasets as dset
import torchvision.transforms as transforms


class StackedMNIST(dset.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, subset_ratio=1):
        super(StackedMNIST, self).__init__(root=root, train=train, transform=transform,
                                           target_transform=target_transform, download=download)

        perm1 = np.random.permutation(len(self.data))
        perm2 = np.random.permutation(len(self.data))
        perm3 = np.random.permutation(len(self.data))
        perm4 = np.random.permutation(len(self.data))
        perm5 = np.random.permutation(len(self.data))
        perm6 = np.random.permutation(len(self.data))
        
        drop_count = int(len(perm1) * (1 - subset_ratio))
        perm1 = perm1[:len(perm1) - drop_count]
        perm2 = perm2[:len(perm2) - drop_count]
        perm3 = perm3[:len(perm3) - drop_count]
        perm4 = perm4[:len(perm4) - drop_count]
        perm5 = perm5[:len(perm5) - drop_count]
        perm6 = perm6[:len(perm6) - drop_count]
        
        index1 = np.hstack([perm1, perm2])
        index2 = np.hstack([perm3, perm4])
        index3 = np.hstack([perm5, perm6])
        
        self.num_images = len(index1)
        
        self.index = [(index1[i], index2[i], index3[i]) for i in range(self.num_images)]


    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        img = np.zeros((28, 28, 3), dtype=np.uint8)
        target = 0
        for i in range(3):
            img_, target_ = self.data[self.index[index][i]], int(self.targets[self.index[index][i]])
            img[:, :, i] = img_
            target += target_ * 10 ** (2 - i)

        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def _data_transforms_stacked_mnist():
    """Get data transforms for cifar10."""
    train_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    valid_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    return train_transform, valid_transform

if __name__ == "__main__":
    train_transform, valid_transform = _data_transforms_stacked_mnist()
    train_dataset = StackedMNIST(root="data", train=True, transform=train_transform, download=True)
    valid_dataset = StackedMNIST(root="data", train=False, transform=valid_transform, download=True)
    print(len(train_dataset), len(valid_dataset))
    print(train_dataset[0])
    print(valid_dataset[0])