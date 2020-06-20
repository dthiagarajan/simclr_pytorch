from torch.utils.data import Dataset
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
