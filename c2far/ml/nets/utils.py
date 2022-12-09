"""Utility functions that would be used by pytorch neural networks.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/nets/utils.py

from torch import nn


def make_extrema_mlps(nextrema_subset, nhidden, extremas):
    """Helper function to create the two internal fully-connected neural
    networks that return the extreme low and extreme high parameters.
    They both have the same sequential MLP structure, with the number
    of units in the hidden layer given as an argument.

    Arguments:

    nextrema_subset: Int, how many features in the extrema subset,
    i.e., how many inputs to this MLP.

    nhidden: Int, the number of units in the hidden layer.

    extremas: Boolean, if True, make the nets, if False, return None for each.

    """
    if extremas:
        net_low = nn.Sequential(nn.Linear(nextrema_subset, nhidden), nn.ReLU(), nn.Linear(nhidden, 1))
        net_high = nn.Sequential(nn.Linear(nextrema_subset, nhidden), nn.ReLU(), nn.Linear(nhidden, 1))
        return net_low, net_high
    return None, None
