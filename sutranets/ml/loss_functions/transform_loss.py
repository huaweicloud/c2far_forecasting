"""Base class for custom loss functions related to loss where we first
transform our outputs to be predicted values in the original domain,
and then apply a standard point-loss-function.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/loss_functions/transform_loss.py
from abc import ABC, abstractmethod
import logging
from torch import nn
from sutranets.ml.loss_functions.constants import IGNORE_INDEX, PointLossFunc
from sutranets.ml.loss_functions.standard_losses import \
    SMAPELoss, NDNormalizer, MPELoss, CoverageLoss, CovWidthLoss, WQLLoss
logger = logging.getLogger("sutranets.ml.loss_functions.transform_loss")
LIST_LOSSES = [PointLossFunc.COVERAGE, PointLossFunc.COV_WIDTH,
               PointLossFunc.WQL]
LOSS_FUNC_DICT = {PointLossFunc.MAE: nn.L1Loss(reduction="mean"),
                  PointLossFunc.MPE: MPELoss(),
                  PointLossFunc.MSE: nn.MSELoss(reduction="mean"),
                  PointLossFunc.SMAPE: SMAPELoss(),
                  PointLossFunc.ND_NORMALIZER: NDNormalizer(),
                  PointLossFunc.COVERAGE: CoverageLoss(),
                  PointLossFunc.COV_WIDTH: CovWidthLoss(),
                  PointLossFunc.WQL: WQLLoss()}


class TransformLoss(ABC):
    """Base class for certain losses (e.g. ValuesLoss): that map back to
    original data domain and compute error there against originals.

    """
    NONFLAT = True

    def __init__(self, pt_loss_func, guesser, target_horizon, *,
                 ignore_index=IGNORE_INDEX):
        self.pt_loss_func = pt_loss_func
        self.guesser = guesser
        self.target_horizon = target_horizon
        if target_horizon is not None:
            hidx, hngen = target_horizon
            if hidx >= hngen:
                raise RuntimeError("Zero-indexed horizon > ngenerated")
        self.ignore_index = ignore_index
        self.loss_func = self.__get_loss_func()

    def __get_loss_func(self):
        """Switch to get the loss function (a function or a callable object).

        """

        loss_func = LOSS_FUNC_DICT.get(self.pt_loss_func)
        if loss_func is None:
            msg = f"Loss function {self.pt_loss_func} not implemented"
            raise RuntimeError(msg)
        return loss_func

    def __filter_include_pts(self, guesses2, include_pts):
        """Given the flattened, transformed guesses and include_pts, filter
        the points that are not at the included points.

        """
        if self.pt_loss_func in LIST_LOSSES:
            guesses2 = [u[include_pts] for u in guesses2]
        else:
            guesses2 = guesses2[include_pts]
        return guesses2

    def _get_info(self):
        """Returns a string that is available to be used in the __str__ method
        of derived classes.

        """
        guesser_info = self.guesser.get_info()
        if self.target_horizon is None:
            info_str = f"{self.pt_loss_func.value}.{guesser_info}"
        else:
            horiz_offset = self.target_horizon[0]
            info_str = f"{self.pt_loss_func.value}." \
                       f"horiz={horiz_offset}.{guesser_info}"
        return info_str

    @abstractmethod
    def __str__(self):
        """Each derived class decides how it wants to be printed, but they can
        use _get_info() in the base class.

        """

    def __get_include_pts(self, targets):
        """Determine and return the "include points": the points where you
        want to compute loss.

        """
        targets = targets[:, :, :1]
        include_pts = (targets != self.ignore_index)
        if self.target_horizon is not None:
            horiz_offset, gsize = self.target_horizon
            nseq = len(include_pts)
            horiz_pt = nseq - gsize + horiz_offset
            include_pts[:horiz_pt] = False
            include_pts[horiz_pt + 1:] = False
        return include_pts

    def __call__(self, outputs, targets, originals, values=None):
        """Converts our outputs to guesses, then computes a standard loss.

        """
        include_pts = self.__get_include_pts(targets)
        include_pts2 = include_pts.reshape(-1)
        originals2 = originals[2:].reshape(-1)
        gold_values2 = originals2[include_pts2]
        guesses2 = self.guesser.get_guesses(outputs, originals, values=values)
        guesses2 = self.__filter_include_pts(guesses2, include_pts2)
        loss = self.loss_func(guesses2, gold_values2)
        num = len(gold_values2)
        return loss, num
