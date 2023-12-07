"""Loss function for outputs (logits) representing the probability
over a binning.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/loss_functions/coarse_cross_entropy_loss.py

import logging
from math import log
import torch
from torch.distributions.pareto import Pareto
from sutranets.ml.loss_functions.constants import IGNORE_INDEX
TOLERANCE = 1e-4
SMOOTHING = 3e-1
TARGET_SHIFT = 1e-2
PARETO_START_CORRECTION = 0.1
logger = logging.getLogger("sutranets.ml.loss_functions.coarse_cross_entropy_loss")


class CoarseCrossEntropyLoss():
    """Differentiable NLL based on the NLL of the coarse and possibly
    extreme components.

    """
    def __init__(self, coarse_cutoffs, *, extremas, ignore_index=IGNORE_INDEX):
        self.ncoarse = len(coarse_cutoffs) + 1
        self.ignore_index = ignore_index
        self.ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="mean")
        self.extremas = extremas
        self.log_width, self.pareto_lo, self.pareto_hi = self.process_coarse_cutoffs(coarse_cutoffs)
        self.softplus = torch.nn.Softplus()

    @staticmethod
    def process_coarse_cutoffs(coarse_cutoffs):
        """Process the coarse cutoffs in order to extract the probability of
        any uniform bin, as well as the high/lo bin boundaries (if we
        are doing extremas).

        """
        pareto_lo = coarse_cutoffs[0] * -1.0
        pareto_hi = coarse_cutoffs[-1]
        width = torch.diff(coarse_cutoffs)
        constant_width = width[0]
        dist = torch.abs(width - constant_width)
        if not (dist < TOLERANCE).all():
            msg = f"Coarse CE Loss not implemented for non-linear cutoffs: {width}, {dist}"
            raise RuntimeError(msg)
        log_width = log(constant_width)
        return log_width, pareto_lo, pareto_hi

    def __get_ex_loss(self, targets, mask, outputs, pareto_start):
        """Compute the *TOTAL* (reduction=sum) Pareto losses on the given
        (non-ignored) targets (via mask), using the output as
        parameter (alpha) inputs, using a Pareto distribution starting
        at pareto_start.

        """
        targets = targets[mask]
        if targets.nelement() == 0:
            return 0.0
        outputs = outputs[mask]
        alphas = self.softplus(outputs.reshape(-1)) + SMOOTHING
        pareto_start = pareto_start.to(alphas.device)
        if pareto_start <= TOLERANCE:
            pareto_correction = PARETO_START_CORRECTION - pareto_start
            pareto_start += pareto_correction  # n.b. thus includes tolerance
            targets += pareto_correction
        targets += TARGET_SHIFT
        try:
            my_pareto = Pareto(pareto_start, alphas)
        except ValueError as excep:
            logger.error("Pareto_start is %s", pareto_start)
            logger.error("alphas are %s", alphas)
            logger.error("outputs after mask are %s", outputs)
            logger.error("mask is %s", mask)
            logger.error("targets are %s", targets)
            logger.exception("Pareto error")
            raise excep
        loss = - my_pareto.log_prob(targets).sum()
        return loss

    def __get_uni_loss(self, ex_lo_mask, ex_hi_mask, coarse_targets):
        """Compute the *TOTAL* (reduction=sum) uniform distribution losses by
        (1) finding all the points that are not extrema (via the ex_lo
        and ex_hi mask), and (2) adding the standard NLL for
        uniforms log(W) loss at those points.  Also return the total
        number of points, for convenience.

        """
        extremas = torch.logical_or(ex_lo_mask, ex_hi_mask)
        nextremas = torch.sum(extremas)
        npts = (coarse_targets != self.ignore_index).sum()
        nuni = npts - nextremas
        uni_loss = nuni * self.log_width
        return uni_loss, npts

    def _compute_density_loss(self, outputs, targets, coarse_targets):
        """Compute the density loss - if not using extremas, it's just the
        log_width, but if we are, then it decomposes into the extreme
        low/high losses and uniform (log_width) losses, which we total
        up and divide by the number of non-ignored points.

        """
        if self.extremas:
            ex_lo_outputs = outputs[:, -2:-1]
            ex_lo_targets = targets[:, -2].reshape(-1)
            ex_lo_mask = torch.logical_or(ex_lo_targets < self.ignore_index - TOLERANCE,
                                          ex_lo_targets > self.ignore_index + TOLERANCE)
            lo_loss = self.__get_ex_loss(ex_lo_targets * -1.0, ex_lo_mask, ex_lo_outputs, self.pareto_lo)
            ex_hi_outputs = outputs[:, -1:]
            ex_hi_targets = targets[:, -1].reshape(-1)
            ex_hi_mask = torch.logical_or(ex_hi_targets < self.ignore_index - TOLERANCE,
                                          ex_hi_targets > self.ignore_index + TOLERANCE)
            hi_loss = self.__get_ex_loss(ex_hi_targets, ex_hi_mask, ex_hi_outputs, self.pareto_hi)
            uni_loss, npts = self.__get_uni_loss(ex_lo_mask, ex_hi_mask, coarse_targets)
            density_loss = (lo_loss + hi_loss + uni_loss) / npts
        else:
            density_loss = self.log_width
        return density_loss

    def __call__(self, outputs, targets):
        """Computes CrossEntropyLoss over coarse and extrema separately, then
        sum them and return loss as a tensor.

        """
        coarse_outputs = outputs[:, :self.ncoarse]
        if self.extremas:
            coarse_targets = targets[:, 0].reshape(-1).long()
        else:
            coarse_targets = targets
        density_loss = self._compute_density_loss(outputs, targets, coarse_targets)
        coarse_loss = self.ce_loss_func(coarse_outputs, coarse_targets)
        return coarse_loss + density_loss

    def __str__(self):
        label = f"<CoarseCE.{self.ncoarse}.extremas={self.extremas}>"
        return label
