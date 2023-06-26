"""Provides loss functions."""
import torch


cel = torch.nn.CrossEntropyLoss()
bce = torch.nn.BCELoss()

def cross_entropy_one_hot(input, target):
    """Gets the cross entropy loss for one hot encoding."""

    _, labels = target.max(dim=1)
    return cel(input, labels)


def binary_cross_entropy_one_hot(input, target):
    """Gets the binary cross entropy loss for one hot encoding."""
    _, labels = input.max(dim=1)
    # casting necessary
    labels = labels.type(torch.LongTensor)
    target = target.type(torch.LongTensor)
    return bce(labels, target)
