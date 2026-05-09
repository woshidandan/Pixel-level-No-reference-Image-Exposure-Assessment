import torch.nn as nn


def freeze_weights(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_weights(module):
    for param in module.parameters():
        param.requires_grad = True


def freeze_bn(module):
    for c in module.children():
        if isinstance(c, nn.BatchNorm2d):
            freeze_weights(c)
        else:
            freeze_bn(c)


def name_size_grad(module):
    for name, param in module.named_parameters():
        print('name ->', name, ', size -> ', param.size(), ', requires_grad->', param.requires_grad)
