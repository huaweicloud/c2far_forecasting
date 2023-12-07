"""Various enumerations/constants that we use in our different loss
functions.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/loss_functions/constants.py
from enum import Enum
IGNORE_INDEX = -100  # PyTorch default


class PointLossFunc(Enum):
    """An enum of error metric types.

    """
    MAE = "MAE"
    MPE = "MPE"
    MSE = "MSE"
    SMAPE = "SMAPE"
    COVERAGE = "Coverage"
    COV_WIDTH = "CoverageWidth"
    WQL = "WQL"
    ND_NORMALIZER = "ND_Normalizer"


class GenStrategy(Enum):
    """An enum of output generation strategies, whether we should pick the
    most likely output (MLE), or the P50, or the mean, etc.

    """
    P50 = "P50"
    GEN_P50 = "gen_p50"
    GEN_COV = "gen_cov"
    GEN_WQL = "gen_wql"
