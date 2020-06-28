""" Trains a torchvision model in the SimCLR formulation using PyTorch Lightning.

Example usage:
python train.py --model_name resnet18 --projection_dim 64 --fast_dev_run True

To run with a single GPU:
python train.py --model_name resnet18 --projection_dim 64 --fast_dev_run True --gpus 1
"""
import argparse
from pytorch_lightning import Trainer

from simclr.model import SimCLRModel
from simclr.mixup_model import SimCLRMixupModel


def get_args():
    """Argument parser running the SimCLR model.

    Returns:
        argparse.Namespace: namespace with all specified arguments
    """
    parser = argparse.ArgumentParser(
        'Train a torchvision model in the SimCLR framework on CIFAR10'
    )
    parser.add_argument('--mixup', action='store_true')
    parser = SimCLRModel.add_model_specific_args(parser)
    parser = SimCLRMixupModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.mixup:
        print(f'Using mixup with training: alpha {args.alpha}, mixup layer {args.mixup_layer}.')
        model = SimCLRMixupModel(
            model_name=args.model_name,
            pretrained=args.pretrained,
            projection_dim=args.projection_dim,
            temperature=args.temperature,
            dataset=args.dataset,
            download=args.download,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            alpha=args.alpha,
            mixup_layer=args.mixup_layer
        )
    else:
        print(f'Using standard model, ignoring mixup flags.')
        model = SimCLRModel(
            model_name=args.model_name,
            pretrained=args.pretrained,
            projection_dim=args.projection_dim,
            temperature=args.temperature,
            dataset=args.dataset,
            download=args.download,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size
        )
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)
