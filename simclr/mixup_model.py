import argparse
import torch.nn as nn

from simclr.data import SimCLRDataset
from simclr.model import SimCLRModel


class SimCLRMixupModel(SimCLRModel):
    """SimCLR mixup training for a generic torchvision model. """

    allowed_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    allowed_datasets = ['CIFAR10', 'CIFAR100', 'STL10', 'SVHN']

    def __init__(
        self, model_name='resnet18', pretrained=True, projection_dim=64, temperature=0.5,
        download=False, dataset='CIFAR10', data_dir='/home/ubuntu/data', alpha=0.4,
        mixup_layer=-1
    ):
        super().__init__(
            model_name, pretrained, projection_dim, temperature, download, dataset, data_dir
        )
        self.model = nn.ModuleList(list(self.model.children()))
        self.alpha = alpha
        self.mixup_layer = mixup_layer
        if self.mixup_layer == -1:
            print(
                f'Warning: no mixup is being done - if you want mixup to be done, '
                f'please specify mixup_layer >= 0.'
            )
        elif self.mixup_layer == 0:
            print(
                f'Mixup is being done on the data. If you want to mixup a particular layer input, '
                f'please specify mixup_layer > 0 corresponding to the index of that layer.'
            )
        elif self.mixup_layer > len(self.model):
            print(
                f'Mixup layer specified is too large, please specify a number in '
                f'[0, {len(self.model)}].'
            )

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.model):
            if i == self.mixup_layer:
                out = SimCLRDataset.mixup(out, alpha=self.alpha)
            out = layer(out)
        out = out.view(x.size(0), -1)

        if self.mixup_layer == len(self.model):
            out = SimCLRDataset.mixup(out, alpha=self.alpha)
        out = self.projection_head(out)
        return out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--alpha', type=float, default=0.4)
        parser.add_argument('--mixup_layer', type=int, default=-1)
        return parser
