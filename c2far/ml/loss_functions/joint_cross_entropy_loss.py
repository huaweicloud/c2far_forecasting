"""Loss function for outputs (logits) representing the joint
probability over two binnings.  We are also given two targets for each
output.  The output uses the NLL (CrossEntropyLoss) over the joint
probability (which is the sum of the NLL for each), plus it
incorporates the (uniform or Pareto) probability density within each
bin.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/loss_functions/joint_cross_entropy_loss.py
import logging
from math import exp, log
import torch
from c2far.ml.loss_functions.coarse_cross_entropy_loss import CoarseCrossEntropyLoss
from c2far.ml.loss_functions.constants import IGNORE_INDEX

logger = logging.getLogger("c2far.ml.loss_functions.joint_cross_entropy_loss")


class JointCrossEntropyLoss(CoarseCrossEntropyLoss):
    """Differentiable NLL based on the NLL of the coarse and fine
    components.  Uses 'mean' reduction, as always in c2far.

    """
    def __init__(self, coarse_cutoffs, nfine, *, extremas, ignore_index=IGNORE_INDEX):
        """We only need the number of coarse bins, as we assume the remainder
        of the logits represent the fine bins

        Arguments:

        coarse_cutoffs: Tensor[Float], the coarse bin boundaries.

        nfine: Int, how many fine bins

        Keyword-only args:

        extremas: Boolean, if True, compute loss specially in the
        extreme bins.

        Optional Arguments:

        ignore_index: Int, a value such that, if target is set to this
        value, we don't include comparisons over those targets in our
        calculation. As in PyTorch, "the loss is averaged over
        non-ignored targets."

        """
        self.ncoarse = len(coarse_cutoffs) + 1
        self.nfine = nfine
        self.ignore_index = ignore_index
        self.ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="mean")
        self.extremas = extremas
        self.log_width, self.pareto_lo, self.pareto_hi = self.process_fine_cutoffs(coarse_cutoffs, nfine)
        self.softplus = torch.nn.Softplus()

    @classmethod
    def process_fine_cutoffs(cls, coarse_cutoffs, nfine):
        """Process the cutoffs (coarse & fine) in order to extract the
        probability of any uniform bin, as well as the high/lo bin
        boundaries (if we are doing extremas).  All this assumes that
        the bins are evenly divided (for now) - if not, raise an
        error.

        """
        log_width, coarse_lo, coarse_hi = cls.process_coarse_cutoffs(coarse_cutoffs)
        width = exp(log_width)
        try:
            fine_width = width / nfine
        except ZeroDivisionError as excep:
            logger.error("nfine can't be zero.")
            raise excep
        fine_hi = coarse_hi + width - fine_width
        fine_lo = coarse_lo + width - fine_width  # note already flipped to +ve
        return log(fine_width), fine_lo, fine_hi

    def __call__(self, outputs, targets):
        """Computes CrossEntropyLoss over the coarse and fine portions
        separately, then sums them and returns loss as a tensor.

        Arguments:

        outputs: tensor: NSEQ x NBATCH x NOUTPUT_DIMS==ncoarse+nfine[+2extremas]

        targets: tensor: NSEQ x NBATCH x 2[+2extremas]

        """
        coarse_outputs = outputs[:, :self.ncoarse]
        coarse_targets = targets[:, 0].reshape(-1)
        fine_outputs = outputs[:, self.ncoarse:self.ncoarse + self.nfine]
        fine_targets = targets[:, 1].reshape(-1)
        if self.extremas:
            coarse_targets = coarse_targets.long()
            fine_targets = fine_targets.long()
        density_loss = self._compute_density_loss(outputs, targets, coarse_targets)
        coarse_loss = self.ce_loss_func(coarse_outputs, coarse_targets)
        fine_loss = self.ce_loss_func(fine_outputs, fine_targets)
        return coarse_loss + fine_loss + density_loss

    def __str__(self):
        """Be more explicit when we're outputting the name (e.g. in directory paths)."""
        label = f"<JointCE.{self.ncoarse}.{self.nfine}.extremas={self.extremas}>"
        return label
