"""Loss function for outputs (logits) representing the probability
over a binning.  We are given one targets for each output.  The output
uses the NLL (CrossEntropyLoss) over the probability, plus it
incorporates the (uniform or Pareto) density within each bin.  The
reasons to use this over vanilla nn.CrossEntropyLoss is that (1) this
function can also incorporate loss from extremas, and (2) this
function normalizes by the size of the bins - i.e., we are doing a
real PDF probability from a piecewise uniform distribution, as opposed
to a softmax.

A note on implementation (of this and the joint/triple): since the
probabilities multiply together, then it's a matter of summing the
log-probabilities, so we can compute the probability within the
non-extrema-bins and the probability within the extreme bins
separately, and just add them to the CrossEntropyLoss.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/loss_functions/coarse_cross_entropy_loss.py

import logging
from math import log
import torch
from torch.distributions.pareto import Pareto
from c2far.ml.loss_functions.constants import IGNORE_INDEX
TOLERANCE = 1e-4
SMOOTHING = 3e-1
TARGET_SHIFT = 1e-2
PARETO_START_CORRECTION = 0.1
logger = logging.getLogger("c2far.ml.loss_functions.coarse_cross_entropy_loss")


class CoarseCrossEntropyLoss():
    """Differentiable NLL based on the NLL of the coarse and possibly
    extreme components.  Uses 'mean' reduction, as always in c2far.

    """
    def __init__(self, coarse_cutoffs, *, extremas, ignore_index=IGNORE_INDEX):
        """Arguments:

        coarse_cutoffs: Tensor[Float], the bin boundaries.

        Keyword-only arguments:

        extremas: Boolean, if True, compute loss specially in the
        extreme bins.

        Optional Arguments:

        ignore_index: Int, a value such that, if target is set to this
        value, we don't include comparisons over those targets in our
        calculation. As in PyTorch, "the loss is averaged over
        non-ignored targets."

        """
        self.ncoarse = len(coarse_cutoffs) + 1
        self.ignore_index = ignore_index
        self.ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="mean")
        self.extremas = extremas
        # **NOTE**: pareto_lo & pareto_hi are not same thing as
        # coarse_lo & coarse_hi: coarse_hi/lo are implicit *ends* of
        # hi/lo bins, whereas pareto_lo/hi are *starts* of these bins:
        self.log_width, self.pareto_lo, self.pareto_hi = self.process_coarse_cutoffs(coarse_cutoffs)
        # Actually only needed for extremas:
        self.softplus = torch.nn.Softplus()

    @staticmethod
    def process_coarse_cutoffs(coarse_cutoffs):
        """Process the coarse cutoffs in order to extract the probability of
        any uniform bin, as well as the high/lo bin boundaries (if we
        are doing extremas).  All this assumes that the bins are
        evenly divided (for now) - if not, raise an error.  Note: also
        used in derived classes.

        Arguments:

        coarse_cutoffs: Tensor[Float], the bin boundaries.

        """
        # We flip lo targets so we have a standard Pareto:
        pareto_lo = coarse_cutoffs[0] * -1.0
        pareto_hi = coarse_cutoffs[-1]
        width = torch.diff(coarse_cutoffs)
        # Take the first one as the width we use for normalization:
        constant_width = width[0]
        # Given we are in a bin, and we are uniform, the prob of each
        # point is 1/W, so actually the negative log-likelihood is
        # log(W), and therefore we return and use that in these bins:
        log_width = log(constant_width)
        return log_width, pareto_lo, pareto_hi

    def __get_ex_loss(self, targets, mask, outputs, pareto_start):
        """Compute the *TOTAL* (reduction=sum) Pareto losses on the given
        (non-ignored) targets (via mask), using the output as
        parameter (alpha) inputs, using a Pareto distribution starting
        at pareto_start.

        Note on pareto_start: it is possible pareto_start will be
        NEGATIVE, e.g., if the last cutoff is negative and we're going
        upwards OR the first cutoff is positive and we're going
        downwards.  In this case, we assume it starts from
        PARETO_START_CORRECTION, and we shift all the targets up
        accordingly.

        """
        targets = targets[mask]
        if targets.nelement() == 0:
            return 0.0
        outputs = outputs[mask]
        # Make sure alphas (1) positive and (2) not infinitesimally small:
        alphas = self.softplus(outputs.reshape(-1)) + SMOOTHING
        pareto_start = pareto_start.to(alphas.device)
        if pareto_start <= TOLERANCE:
            pareto_correction = PARETO_START_CORRECTION - pareto_start
            pareto_start += pareto_correction  # n.b. thus includes tolerance
            targets += pareto_correction
        # Nudge targets up - unless we do this, we can get "argument
        # must be within support" ValueErrors in log_prob below:
        targets += TARGET_SHIFT
        try:
            my_pareto = Pareto(pareto_start, alphas)
        except ValueError as excep:
            logger.exception("Pareto error %s %s %s %s %s",
                             pareto_start, alphas, outputs, mask, targets)
            raise excep
        # Note we sum here, normalize to mean later:
        loss = - my_pareto.log_prob(targets).sum()
        return loss

    def __get_uni_loss(self, ex_lo_mask, ex_hi_mask, coarse_targets):
        """Compute the *TOTAL* (reduction=sum) uniform distribution losses by
        (1) finding all the points that are not extrema (via the ex_lo
        and ex_hi mask), and (2) adding the standard NLL for
        uniforms log(W) loss at those points.  Also return the total
        number of points, for convenience.

        Arguments:

        coarse_targets: Tensor[Long], the coarse targets, used purely
        to determine which points are IGNORED points.

        """
        extremas = torch.logical_or(ex_lo_mask, ex_hi_mask)
        nextremas = torch.sum(extremas)
        npts = (coarse_targets != self.ignore_index).sum()
        nuni = npts - nextremas
        # We suffer the log_width NLL on each of these non-extrema points:
        uni_loss = nuni * self.log_width
        return uni_loss, npts

    def _compute_density_loss(self, outputs, targets, coarse_targets):
        """Function to compute the density loss - if not using extremas, it's
        just the log_width, but if we are, then it decomposes into the
        extreme low/high losses and uniform (log_width) losses, which
        we total up and divide by the number of non-ignored points.

        Arguments:

        coarse_targets: Tensor[Long], the coarse targets, used purely
        to determine which points are IGNORED points.

        """
        if self.extremas:
            # density loss is: (1) total loss on extreme lows + (2)
            # total loss on extreme highs + (3) total loss on
            # uniforms, all divided by num. points (mean reduction)
            ex_lo_outputs = outputs[:, -2:-1]
            ex_lo_targets = targets[:, -2].reshape(-1)
            ex_lo_mask = torch.logical_or(ex_lo_targets < self.ignore_index - TOLERANCE,
                                          ex_lo_targets > self.ignore_index + TOLERANCE)
            # Note we flip the low targets so we have a standard Pareto:
            lo_loss = self.__get_ex_loss(ex_lo_targets * -1.0, ex_lo_mask, ex_lo_outputs, self.pareto_lo)
            ex_hi_outputs = outputs[:, -1:]
            ex_hi_targets = targets[:, -1].reshape(-1)
            ex_hi_mask = torch.logical_or(ex_hi_targets < self.ignore_index - TOLERANCE,
                                          ex_hi_targets > self.ignore_index + TOLERANCE)
            hi_loss = self.__get_ex_loss(ex_hi_targets, ex_hi_mask, ex_hi_outputs, self.pareto_hi)
            uni_loss, npts = self.__get_uni_loss(ex_lo_mask, ex_hi_mask, coarse_targets)
            density_loss = (lo_loss + hi_loss + uni_loss) / npts
        else:
            # The average (conditional) loss on every point is the
            # same as the loss on a single point: log(W) - note this
            # is true whether or not any targets are ignored (unless
            # there are zero non-ignored, but then we'll contribute
            # zero to the running total).
            density_loss = self.log_width
        return density_loss

    def __call__(self, outputs, targets):
        """Computes CrossEntropyLoss over coarse and extrema separately, then
        sum them and return loss as a tensor.

        Arguments:

        outputs: tensor: NSEQ x NBATCH x NCOARSE_DIMS==ncoarse[+2extremas]

        targets: tensor: NSEQ x NBATCH x 1[+2extremas]

        """
        # Do the coarse and within-bin losses separately, then add:
        coarse_outputs = outputs[:, :self.ncoarse]  # could be all of them, if no extremas
        if self.extremas:
            # When we mix extreme vals + longs together in targets, it
            # becomes a float tensor, and we need to convert back now:
            coarse_targets = targets[:, 0].reshape(-1).long()
        else:
            coarse_targets = targets
        density_loss = self._compute_density_loss(outputs, targets, coarse_targets)
        coarse_loss = self.ce_loss_func(coarse_outputs, coarse_targets)
        return coarse_loss + density_loss

    def __str__(self):
        """Be more explicit when we're outputting the name (e.g. in directory paths)."""
        label = f"<CoarseCE.{self.ncoarse}.extremas={self.extremas}>"
        return label
