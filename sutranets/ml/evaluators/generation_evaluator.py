"""A subclass of Evaluator but that does multi-step-ahead generation.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/generation_evaluator.py
from sutranets.ml.evaluators.evaluator import Evaluator


class GenerationEvaluator(Evaluator):
    """Class to help with testing of multi-step-ahead generation."""
    LABEL = "GenerationEvaluator"

    def __init__(self, batch_pred_mgr, device):
        super().__init__(device)
        self.batch_pred_mgr = batch_pred_mgr

    def _make_outputs(self, inputs, targets, originals):
        """Override with our generation-specific forward pass.

        """
        pred_inputs = self.batch_pred_mgr.prep_inputs(originals)
        ptiles = self.batch_pred_mgr(pred_inputs)
        pred_outputs = self.batch_pred_mgr.prep_outputs(ptiles)
        return pred_outputs

    def __str__(self):
        return f"{self.LABEL}.{self.batch_pred_mgr}"
