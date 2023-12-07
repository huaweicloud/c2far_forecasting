"""Negative-log-likelihood loss function for outputs (logits)
representing the probability of the outputs for multiple different
subseries.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/loss_functions/multivariate_loss.py
from sutranets.ml.evaluators.evaluator import Evaluator
from sutranets.ml.loss_functions.constants import IGNORE_INDEX


class MultivariateLoss():
    """Differentiable NLL based on the NLL of all the sub-series within
    the multivariate series.

    """
    NONFLAT = True

    def __init__(self, losses):
        self.losses = losses

    def __call__(self, outputs, targets, originals, coarses):
        """Computes the appropriate log-losses over each sub-series separately,
        then sum them and returns loss as a tensor.

        """
        targets_lst = targets.get_lst_of_tensors()
        total_loss = 0
        total_num = 0
        if len(self.losses) != len(outputs) != len(targets_lst):
            raise RuntimeError("Mismatching losses/outputs/targets")
        for loss, one_outputs, one_targets in zip(self.losses, outputs, targets_lst):
            flat_outputs, flat_targets = Evaluator.flatten_tensors(one_outputs, one_targets)
            total_loss += loss(flat_outputs, flat_targets)
            total_num += (one_targets[:, :, 0] != IGNORE_INDEX).sum()
        return total_loss, total_num

    def __str__(self):
        component_loss = str(self.losses[0])
        label = f"<MultivLoss.{component_loss}>"
        return label
