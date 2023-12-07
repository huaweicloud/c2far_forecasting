"""Loss function for outputs (logits) representing the joint
probability over two binnings.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/loss_functions/joint_cross_entropy_loss.py

from math import exp, log
import torch
from sutranets.ml.loss_functions.coarse_cross_entropy_loss import CoarseCrossEntropyLoss
from sutranets.ml.loss_functions.constants import IGNORE_INDEX


class JointCrossEntropyLoss(CoarseCrossEntropyLoss):
    """Differentiable NLL based on the NLL of the coarse and fine
    components.

    """
    def __init__(self, coarse_cutoffs, nfine, *, extremas, ignore_index=IGNORE_INDEX):
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
        boundaries (if we are doing extremas).

        """
        log_width, coarse_lo, coarse_hi = cls.process_coarse_cutoffs(coarse_cutoffs)
        width = exp(log_width)
        fine_width = width / nfine
        fine_hi = coarse_hi + width - fine_width
        fine_lo = coarse_lo + width - fine_width
        return log(fine_width), fine_lo, fine_hi

    def __call__(self, outputs, targets):
        """Computes CrossEntropyLoss over the coarse and fine portions
        separately, then sums them and returns loss as a tensor.

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
        label = f"<JointCE.{self.ncoarse}.{self.nfine}.extremas={self.extremas}>"
        return label
