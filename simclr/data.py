import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms


class SimCLRDataset(Dataset):
    def __init__(self, dataset):
        """Initialize a wrapper of a generic image classification dataset for SimCLR training.

        Args:
            dataset (torch.utils.data.Dataset): an image PyTorch dataset - when iterating over it
                it should return something of the form (image) or (image, label).
        """
        self.dataset = dataset

    def __getitem__(self, index):
        dataset_item = self.dataset[index]
        if type(dataset_item) is tuple:
            image = dataset_item[0]
        else:
            image = dataset_item
        return image, image

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def mixup(x, alpha=0.4):
        batch_size = x.size()[0] // 2
        if alpha > 0:
            lam = np.random.beta(alpha, alpha, batch_size)
            lam = np.concatenate(
                [lam[:, None], 1 - lam[:, None]], 1
            ).max(1)[:, None, None, None]
            lam = torch.from_numpy(lam).float()
            if torch.cuda.is_available():
                lam = lam.cuda()
        else:
            lam = 1.
        # This is SimCLR specific - we want to use the same mixing for the augmented pairs
        lam = torch.cat([lam, lam])
        index = torch.randperm(batch_size)
        # This is SimCLR specific - we want to use the same permutation on the augmented pairs
        index = torch.cat([index, batch_size + index])
        if torch.cuda.is_available():
            index = index.cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]

        return mixed_x, lam


def imagenet_normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_train_transforms(size=224, color_jitter_prob=0.8, grayscale_prob=0.2):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    return transforms.Compose([
        transforms.RandomResizedCrop(size=(size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=color_jitter_prob),
        transforms.RandomGrayscale(p=grayscale_prob),
        transforms.ToTensor(),
        imagenet_normalize_transform()
    ])


def get_val_transforms(size=224):
    return transforms.Compose([
        transforms.Resize(size=(size, size)),
        transforms.ToTensor(),
        imagenet_normalize_transform()
    ])


CIFAR10 = datasets.CIFAR10
CIFAR100 = datasets.CIFAR100


def STL10(root, train=True, download=False, transform=None):
    if train:
        dataset_class = 'train+unlabeled'
    else:
        dataset_class = 'test'
    print(f'Retrieving/downloading {dataset_class} set of STL10 dataset.')
    return datasets.STL10(
        root, split=dataset_class, download=download, transform=transform
    )


def SVHN(root, train=True, download=False, transform=None):
    if train:
        dataset_class = 'train'
    else:
        dataset_class = 'test'
    print(f'Retrieving/downloading {dataset_class} set of SVHN dataset.')
    return datasets.SVHN(
        root, split=dataset_class, download=download, transform=transform
    )
