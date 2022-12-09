"""Base class for custom loss functions related to loss where we first
transform our outputs to be predicted values in the original domain,
and then apply a standard point-loss-function.  The generation-based
ones are not differentiable, but the logit-based ones should be.
Should all be capable of working on batches.

The transforming part is done by a "guesser" function, which takes the
logits (outputs) of the LSTM and returns the values that we can
compare to the gold values in the original domain.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/loss_functions/transform_loss.py

from abc import ABC, abstractmethod
import logging
from torch import nn
from c2far.ml.loss_functions.constants import IGNORE_INDEX, PointLossFunc
from c2far.ml.loss_functions.standard_losses import \
    NDNormalizer, CoverageLoss, CovWidthLoss, WQLLoss
logger = logging.getLogger("c2far.ml.loss_functions.transform_loss")
LIST_LOSSES = [PointLossFunc.COVERAGE, PointLossFunc.COV_WIDTH,
               PointLossFunc.WQL]
LOSS_FUNC_DICT = {PointLossFunc.MAE: nn.L1Loss(reduction="mean"),
                  PointLossFunc.ND_NORMALIZER: NDNormalizer(),
                  PointLossFunc.COVERAGE: CoverageLoss(),
                  PointLossFunc.COV_WIDTH: CovWidthLoss(),
                  PointLossFunc.WQL: WQLLoss()}


class TransformLoss(ABC):
    """Base class for certain losses (e.g. ValuesLoss): that map back to
    original data domain and compute error there against originals.

    """
    def __init__(self, pt_loss_func, guesser, *,
                 ignore_index=IGNORE_INDEX):
        """Store the key strategies for what kinds of error to get, and how to
        generate the predictions.

        Arguments:

        pt_loss_func: Enum, the "point" loss: whether we should use
        MSE, MASE, MAPE, etc.

        guesser: Guesser, an object whose get_guesses we call on the
        outputs in order to transform them into something we can
        compare with the loss.

        ignore_index: Int, which index should be ignored in the loss
        functions.

        """
        self.pt_loss_func = pt_loss_func
        self.guesser = guesser
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
        info_str = f"{self.pt_loss_func.value}.{guesser_info}"
        return info_str

    @abstractmethod
    def __str__(self):
        """Each derived class decides how it wants to be printed, but they can
        use _get_info() in the base class.

        """

    def __get_include_pts(self, targets):
        """Determine and returnd the "include points": the points where you
        want to compute loss.  This now happens in original space.

        """
        # If we are doing joint coarse/fine targets, they get
        # ignore_index at the same points, so just use coarse (safe if
        # only doing coarse as well):
        targets = targets[:, :, :1]
        include_pts = (targets != self.ignore_index)
        return include_pts

    def __call__(self, outputs, targets, originals, values=None):
        """Generic code that converts our outputs to guesses, then computes a
        standard loss.

        Arguments:

        outputs: Tensor: NSEQ x NBATCH x NBINS, the logits for each
        bin, for each element of each sequence of each tensor.

        targets: Tensor: NSEQ x NBATCH x 1, the index of the true bin
        for each element of each sequence of each tensor.  We actually
        only use this to determine which points to include in the loss
        calculation (looking for != IGNORE_INDEX)

        originals: Tensor: NSEQ+2 x NBATCH x 1, the original values
        for each batch of sequences, which we compare our transformed
        predictions to at the included points.

        values: Tensor: NSEQ+2 x NBATCH x 1, the encoded values for
        each batch of sequences.  Can pass None if not using certain
        intra-bin-decoding methods.

        """
        include_pts = self.__get_include_pts(targets)
        include_pts2 = include_pts.reshape(-1)
        originals2 = originals[2:].reshape(-1)
        gold_values2 = originals2[include_pts2]
        guesses2 = self.guesser.get_guesses(outputs, originals, values=values)
        guesses2 = self.__filter_include_pts(guesses2, include_pts2)
        loss = self.loss_func(guesses2, gold_values2)
        num = len(gold_values2)  # only includes the include_pts
        return loss, num
