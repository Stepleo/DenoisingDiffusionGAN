
import numpy as np
from PIL import Image
import torchvision.datasets as dset
import torchvision.transforms as transforms

class StackedMNIST2D(dset.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,subset_ratio=1):
        super(StackedMNIST2D, self).__init__(root=root, train=train, transform=transform,
                                             target_transform=target_transform, download=download)

        # Create four random permutations
        perm1 = np.random.permutation(len(self.data))
        perm2 = np.random.permutation(len(self.data))
        perm3 = np.random.permutation(len(self.data))
        perm4 = np.random.permutation(len(self.data))

        # Randomly drop 50% of the indices from each permutation
        drop_count = int(len(perm1) * (1 - subset_ratio))
        perm1 = perm1[:len(perm1) - drop_count]
        perm2 = perm2[:len(perm2) - drop_count]
        perm3 = perm3[:len(perm3) - drop_count]
        perm4 = perm4[:len(perm4) - drop_count]

        # Concatenate the remaining indices to form index1 and index2
        index1 = np.hstack([perm1, perm2])
        index2 = np.hstack([perm3, perm4])

        # Generate pairs of indices
        self.num_images = len(index1)
        self.index = [(index1[i], index2[i]) for i in range(self.num_images)]

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        img = np.zeros((28, 28, 2), dtype=np.uint8)
        target = 0
        for i in range(2):
            img_, target_ = self.data[self.index[index][i]], int(self.targets[self.index[index][i]])
            img[:, :, i] = img_
            target += target_ * 10 ** (1 - i)

        img = Image.fromarray(img, mode="LA")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def _data_transforms_stacked_mnist_2d():
    train_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5), (0.5, 0.5))
    ])

    valid_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5), (0.5, 0.5))
    ])

    return train_transform, valid_transform

if __name__ == "__main__":
    train_transform, valid_transform = _data_transforms_stacked_mnist_2d()
    train_dataset = StackedMNIST2D(root="data", train=True, transform=train_transform, download=True)
    valid_dataset = StackedMNIST2D(root="data", train=False, transform=valid_transform, download=True)
    print(len(train_dataset), len(valid_dataset))
    print(train_dataset[0])
    print(valid_dataset[0])