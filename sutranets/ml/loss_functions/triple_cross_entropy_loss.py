"""Loss function for outputs (logits) representing the triple joint
probability over three binnings.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/loss_functions/triple_cross_entropy_loss.py
from math import exp, log
import torch
from sutranets.ml.loss_functions.constants import IGNORE_INDEX
from sutranets.ml.loss_functions.joint_cross_entropy_loss import JointCrossEntropyLoss


class TripleCrossEntropyLoss(JointCrossEntropyLoss):
    """Differentiable NLL based on the NLL of the coarse, fine, and fine2
    components.

    """
    def __init__(self, coarse_cutoffs, nfine, nfine2, *, extremas,
                 ignore_index=IGNORE_INDEX):
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
        fine2_width = width / nfine2
        fine2_hi = fine_hi + width - fine2_width
        fine2_lo = fine_lo + width - fine2_width
        return log(fine2_width), fine2_lo, fine2_hi

    def __call__(self, outputs, targets):
        """Computes CrossEntropyLoss over the coarse, fine, and fine2 portions
        separately, then sums them and returns loss as a tensor.

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
        label = f"<TripleCE.{self.ncoarse}.{self.nfine}.{self.nfine2}.extremas={self.extremas}>"
        return label
