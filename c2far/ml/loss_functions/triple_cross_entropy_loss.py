"""Loss function for outputs (logits) representing the triple joint
probability over three binnings.  We are also given three targets for
each output.  The output uses the NLL (CrossEntropyLoss) over the
chain-rule probability (which is the sum of the NLL for each), plus it
incorporates the (uniform or Pareto) probability density within each
bin.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/loss_functions/triple_cross_entropy_loss.py
import logging
from math import exp, log
import torch
from c2far.ml.loss_functions.constants import IGNORE_INDEX
from c2far.ml.loss_functions.joint_cross_entropy_loss import JointCrossEntropyLoss

logger = logging.getLogger("c2far.ml.loss_functions.triple_cross_entropy_loss")


class TripleCrossEntropyLoss(JointCrossEntropyLoss):
    """Differentiable NLL based on the NLL of the coarse, fine, and fine2
    components.  Uses 'mean' reduction, as always in c2far.

    """
    def __init__(self, coarse_cutoffs, nfine, nfine2, *, extremas,
                 ignore_index=IGNORE_INDEX):
        """We need the number of coarse and number of fine bins to infer the
        coarse/fine/fine2 splitting (but not nfine2).

        Arguments:

        coarse_cutoffs: Tensor[Float], the coarse bin boundaries.

        nfine: Int, the number of fine bins in the logits.

        nfine2: Int, the number of fine2 bins in the logits.

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
        self.nfine2 = nfine2
        self.ignore_index = ignore_index
        self.ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="mean")
        self.extremas = extremas
        self.log_width, self.pareto_lo, self.pareto_hi = self.process_fine2_cutoffs(
            coarse_cutoffs, nfine, nfine2)
        self.softplus = torch.nn.Softplus()

    @classmethod
    def process_fine2_cutoffs(cls, coarse_cutoffs, nfine, nfine2):
        """Process the cutoffs (coarse & fine & fine2) in order to extract the
        probability of any uniform bin, as well as the high/lo bin
        boundaries (if we are doing extremas).

        """
        log_width, fine_lo, fine_hi = cls.process_fine_cutoffs(coarse_cutoffs, nfine)
        width = exp(log_width)
        try:
            fine2_width = width / nfine2
        except ZeroDivisionError as excep:
            logger.error("nfine2 can't be zero.")
            raise excep
        fine2_hi = fine_hi + width - fine2_width
        fine2_lo = fine_lo + width - fine2_width  # note already flipped to +ve
        return log(fine2_width), fine2_lo, fine2_hi

    def __call__(self, outputs, targets):
        """Computes CrossEntropyLoss over the coarse, fine, and fine2 portions
        separately, then sums them and returns loss as a tensor.

        Arguments:

        outputs: tensor: NSEQ x NBATCH x NOUTPUT_DIMS==ncoarse+nfine+nfine2[+2extremas]

        targets: tensor: NSEQ x NBATCH x 3[+2extremas]

        """
        coarse_targets = targets[:, 0].reshape(-1)
        coarse_outputs = outputs[:, :self.ncoarse]
        fine_targets = targets[:, 1].reshape(-1)
        fine_outputs = outputs[:, self.ncoarse:self.ncoarse + self.nfine]
        fine2_targets = targets[:, 2].reshape(-1)
        fine2_outputs = outputs[:, self.ncoarse + self.nfine:self.ncoarse + self.nfine + self.nfine2]
        if self.extremas:
            coarse_targets = coarse_targets.long()
            fine_targets = fine_targets.long()
            fine2_targets = fine2_targets.long()
        density_loss = self._compute_density_loss(outputs, targets, coarse_targets)
        coarse_loss = self.ce_loss_func(coarse_outputs, coarse_targets)
        fine_loss = self.ce_loss_func(fine_outputs, fine_targets)
        fine2_loss = self.ce_loss_func(fine2_outputs, fine2_targets)
        return coarse_loss + fine_loss + fine2_loss + density_loss

    def __str__(self):
        """Be more explicit when we're outputting the name (e.g. in directory paths)."""
        label = f"<TripleCE.{self.ncoarse}.{self.nfine}.{self.nfine2}.extremas={self.extremas}>"
        return label
