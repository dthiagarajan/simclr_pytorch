import argparse
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

from .datasets import get_train_transforms, get_val_transforms, SimCLRDataset
from .loss import NTXEntCriterion


class SimCLRModel(LightningModule):
    """SimCLR training network for a generic torchvision model (restricted to `allowed_models`). """

    allowed_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

    def __init__(
        self, model_name='resnet18', pretrained=True, projection_dim=64, temperature=0.5,
        download=False, data_dir='/home/ubuntu/data'
    ):
        super().__init__()
        assert model_name in self.allowed_models, f"Please pick one of: {self.allowed_models}"
        layers = list(getattr(torchvision.models, model_name)(pretrained=pretrained).children())
        self.model = nn.Sequential(*layers[:-1])
        self.projection_head = nn.Linear(layers[-1].in_features, projection_dim)
        self.loss = NTXEntCriterion(temperature=temperature)
        self.download = download
        self.data_dir = data_dir
        self.save_hyperparameters()

    def forward(self, x):
        out = self.model(x)
        out = out.view(x.size(0), -1)
        out = self.projection_head(out)
        return out

    def training_step(self, batch, batch_idx):
        projections = self(batch)
        loss = self.loss(projections)  # Change to SimCLR Loss
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        projections = self(batch)
        loss = self.loss(projections)
        tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': 0.00001},
            {'params': self.projection_head.parameters(), 'lr': 0.001}
        ])

    def prepare_data(self):
        train_transforms, val_transforms = get_train_transforms(), get_val_transforms()
        train_dataset = torchvision.datasets.CIFAR10(
            self.data_dir, train=True, download=self.download, transform=train_transforms
        )
        self.train_dataset = SimCLRDataset(train_dataset)
        val_dataset = torchvision.datasets.CIFAR10(
            self.data_dir, train=False, download=self.download, transform=val_transforms
        )
        self.val_dataset = SimCLRDataset(val_dataset)

    def collate_fn(self, batch):
        return torch.cat([torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=32, num_workers=0, shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=32, num_workers=0, shuffle=False,
            collate_fn=self.collate_fn
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='resnet18')
        parser.add_argument('--pretrained', type=bool, default='True')
        parser.add_argument('--projection_dim', type=int, default=64)
        parser.add_argument('--temperature', type=float, default=0.5)
        parser.add_argument('--download', action='store_true')
        parser.add_argument('--data_dir', type=str, default='/home/ubuntu/data')
        return parser
