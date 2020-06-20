# SimCLR with PyTorch Lightning

Implements the SimCLR formulation for any `torchvision` model in PyTorch Lightning. Heavily inspired from [another SimCLR implementation on Github](https://github.com/Spijkervet/SimCLR/).

Notes:
* Currently only allows ResNet models
* Currently only trains on CIFAR10
* Will be updated to include
    * Training on any torchvision dataset
    * Including mixup training, manifold mixup training
    * Tensorboard dev logs of full training runs
