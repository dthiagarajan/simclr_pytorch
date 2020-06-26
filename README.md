# SimCLR with PyTorch Lightning

Implements the SimCLR formulation for any `torchvision` model in PyTorch Lightning. Heavily inspired from [another SimCLR implementation on Github](https://github.com/Spijkervet/SimCLR/).

Allowed datasets:
* CIFAR10
* CIFAR100
* STL10
* SVHN

Allowed models:
* ResNet 18
* ResNet 34
* ResNet 50
* ResNet 101
* ResNet 152

Capable of:
* Mixup training (on the data or on the input to any layer, including the projection head)

Will be updated to include
* Tensorboard dev logs of full training runs (code has only been tested using `fast_dev_run`.)
* Exploring more of what PyTorch Lightning offers for a later blog post
