"""Various enumerations/constants that we use in our different loss
functions.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/loss_functions/constants.py
from enum import Enum
IGNORE_INDEX = -100  # PyTorch default, but be explicit


class PointLossFunc(Enum):
    """An enum of error types, whether we should use MSE, MASE, MAPE,
    etc. Note on terminology: while these are methods to compute loss
    when a given value (y) materializes (y is a point value), they do
    not necessarily require a point *prediction* - e.g. WQL is on all
    the returned qtiles.

    """
    MAE = "MAE"
    COVERAGE = "Coverage"
    COV_WIDTH = "CoverageWidth"
    WQL = "WQL"
    # Special one that ONLY looks at y - gets its average:
    ND_NORMALIZER = "ND_Normalizer"


class GenStrategy(Enum):
    """An enum of output generation strategies.

    For weighted quantile loss as a point loss function, you use
    GEN_WQL as a strategy.

    Otherwise, when running on GENERATED output, you would use GEN_P50
    or GEN_COV.

    """
    GEN_P50 = "gen_p50"
    GEN_COV = "gen_cov"
    GEN_WQL = "gen_wql"
