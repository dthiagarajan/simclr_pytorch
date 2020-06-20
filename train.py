""" Trains a torchvision model in the SimCLR formulation using PyTorch Lightning.

Example usage:
python train.py --model_name resnet18 --projection_dim 64 --fast_dev_run True
"""
import argparse
from pytorch_lightning import Trainer

from simclr.model import SimCLRModel


def get_args():
    """Argument parser running the SimCLR model.

    Returns:
        argparse.Namespace: namespace with all specified arguments
    """
    parser = argparse.ArgumentParser(
        'Train a torchvision model in the SimCLR framework on CIFAR10'
    )
    parser = SimCLRModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    model = SimCLRModel(
        model_name=args.model_name,
        pretrained=args.pretrained,
        projection_dim=args.projection_dim,
        temperature=args.temperature,
        download=args.download,
        data_dir=args.data_dir,
    )
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)
