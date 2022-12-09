"""Constants that can be used by modules doing ML work.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/constants.py

from enum import Enum
import torch

# Note, some argument-related defaults are also in add_args.py.

INPUT_DTYPE = torch.float32


class ProbError(Exception):
    """Typically raised from torch.multinomial when probability tensor
    contains either `inf`, `nan` or element < 0.  We take this to be a
    sign that these are bad parameters (e.g. LR is too high).

    """


class ExampleKeys(Enum):
    """Keys we can use to extract different values from an example."""
    INPUT = "input"
    TARGET = "target"
    ORIGINALS = "originals"
    COARSES = "coarses"


class BinStrategy(Enum):
    """Defines how we create the coarse cutoffs.  Right now, we only have
    the one strategy.

    """
    LINEAR = "linear"


class IntraBinDecoder(Enum):
    """An enum of methods for grabbing a value from WITHIN a bin (after
    the net has picked the bin).

    """
    UNIFORM = "uniform"


class ConfidenceFlags(Enum):
    """Special values of the confidence parameter signalling special
    generation of output.

    """
    WQL_QTILES = -1
