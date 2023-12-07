"""Constants that can be used by modules doing ML work.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/constants.py

from enum import Enum
import torch

INPUT_DTYPE = torch.float32
DUMMY_BIN = 0


class ProbError(Exception):
    """Typically raised from torch.multinomial when probability tensor
    contains either `inf`, `nan` or element < 0.  We take this to be a
    sign that these are bad parameters (e.g. LR is too high).

    """


class ExampleKeys(Enum):
    """Keys we can use to extract different values from an example."""
    INPUT = "input"
    TARGET = "target"
    META = "meta"
    ORIGINALS = "originals"
    COARSES = "coarses"
    FINES = "fines"
    FINES2 = "fines2"


class BinStrategy(Enum):
    """Defines how we create the coarse cutoffs.

    """
    LINEAR = "linear"


class IntraBinDecoder(Enum):
    """An enum of methods for grabbing a value from WITHIN a bin (after
    the net has picked the bin).

    """
    UNIFORM = "uniform"


class ConfidenceFlags(Enum):
    """Known values for confidence that trigger special handling."""
    WQL_QTILES = -1
