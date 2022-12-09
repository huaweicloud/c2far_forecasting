"""Custom loss function that computes the Gaussian NLL, using the
PyTorch function, but with our work to transform the outputs into the
form we need, and we also watch here for IGNORE_INDEX targets.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/loss_functions/gauss_loss.py

import torch
from c2far.ml.loss_functions.constants import IGNORE_INDEX
TOLERANCE = 1e-4


class GaussLoss():
    """Callable class for computing the NLL loss under a Gaussian."""
    def __init__(self, *, ignore_index=IGNORE_INDEX):
        """Init internal callables with default parameters.

        Optional Arguments:

        ignore_index: Int, a value such that, if target is set to this
        value, we don't include comparisons over those targets in our
        calculation. As in PyTorch, "the loss is averaged over
        non-ignored targets."

        """
        self.softplus = torch.nn.Softplus()
        self.gaussian_nll_loss = torch.nn.GaussianNLLLoss(full=True)
        self.ignore_index = ignore_index

    def __call__(self, outputs, targets):
        """The targets are treated as samples from Gaussian distributions with
        expectations and variances given via the outputs.

        Arguments:

        outputs: tensor: NSEQ x NBATCH x 2  # one mean/var for each

        targets: tensor: NSEQ x NBATCH x 1 (real-values)

        ignore_index: Int, a value such that, if target is set to this
        value, we don't include comparisons over those targets in our
        calculation. As in PyTorch, "the loss is averaged over
        non-ignored targets"

        """
        targets = targets.reshape(-1)
        outputs = outputs.reshape(-1, 2)
        include_pts = torch.logical_or(targets < self.ignore_index - TOLERANCE,
                                       targets > self.ignore_index + TOLERANCE)
        new_targets = targets[include_pts]
        new_outputs = outputs[include_pts]
        means = new_outputs[:, 0]
        variances = self.softplus(new_outputs[:, 1])
        loss = self.gaussian_nll_loss(means, new_targets, variances)
        # Note this is already an average over the losses on each
        # point (reduction="mean").
        return loss

    def __str__(self):
        """Be more explicit when we're outputting the name, e.g. in DIRs."""
        label = "GaussLoss.default"
        return label
