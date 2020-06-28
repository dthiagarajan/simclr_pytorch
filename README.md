# SimCLR with PyTorch Lightning

## Overview
Implements the SimCLR formulation for any `torchvision` model in PyTorch Lightning. Heavily inspired from [another SimCLR implementation on Github](https://github.com/Spijkervet/SimCLR/).

## Restrictions/capabilities
### Allowed datasets:
* CIFAR10
* CIFAR100
* STL10
* SVHN

### Allowed models:
* ResNet 18
* ResNet 34
* ResNet 50
* ResNet 101
* ResNet 152

### Capabilities:
* Mixup training (on the data or on the input to any layer, including the projection head)

## Experiments
You can find example training runs [here](https://tensorboard.dev/experiment/NVRvIwOVSl6uqKGUoTGCkQ/). The runs included there are training ResNet18 with the SimCLR formulation on CIFAR10 for about 10 epochs. Specifically:
* `version_0` corresponds to a standard SimCLR training run
* `version_1` corresponds to a SimCLR training run with mixup on the data
* `version_2` corresponds to a SIMCLR training run with mixup on one of the hidden activations


## Future Work
* Tensorboard dev logs of full training runs (code has only been tested using `fast_dev_run`.)
* Exploring more of what PyTorch Lightning offers for a later blog post
