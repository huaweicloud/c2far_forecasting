"""A subclass of Evaluator but that does multi-step-ahead generation.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/evaluators/generation_evaluator.py

import torch
from c2far.ml.evaluators.evaluator import Evaluator


class GenerationEvaluator(Evaluator):
    """Class to help with testing of multi-step-ahead generation."""
    LABEL = "GenerationEvaluator"

    def __init__(self, batch_pred, device):
        """Pass in the batch predictor and the device (torch.device, not the
        string).

        Arguments:

        batch_pred: BatchPredictor, which we call to generate
        predictions.

        device: torch.device (not the string), onto which we place the
        tensors.

        """
        super().__init__(device)
        self.batch_pred = batch_pred

    def _make_outputs(self, inputs, targets, originals):
        """Override the make_outputs part with our generation-specific forward
        pass.  I.e., no longer do one-step-ahead logit prediction, but
        rather return quantiles over the whole prediction window.

        We do this in concert with developing a new criterion, which
        can give us the metrics we want over the future.

        Returns:

        outputs, Tensor: NSEQ x NBATCH x N, where the last GSIZE of
        the NSEQ dimension has the prediction.  N may be 3 if we have
        confidence > 0 and are doing lows, p50s, highs.  Or it may be
        9 if we are doing all the levels for WQL.

        """
        # We already have inputs, targets, and values (genmode=False
        # in the client that calls this). But we ditch all that and
        # just do multi-step generation using originals, because then
        # we can just use the batch_predictor (genmode=True):
        csize, gsize = self.batch_pred.get_csize_gsize()
        if len(originals) != csize + gsize + 2:
            raise RuntimeError("Using originals that do not match given csize/gsize")
        originals = originals[:csize + 2]
        originals = originals.squeeze(2).transpose(0, 1)
        ptiles = self.batch_pred(originals)
        preds = torch.stack(ptiles, 2)
        # These are NBATCH x GSIZE x N (e.g. N=3 for lows, p50s,
        # highs, N=9 when doing WQL). Let's transpose so we have GSIZE x NBATCH x N.
        preds = preds.transpose(0, 1)
        # Note these only have PREDICTIONS. But eval code expects the
        # whole sequence. So let's stick the preds at the end of this:
        my_gsize, nbatch, nptiles = preds.shape
        nseq = csize + gsize
        outputs = torch.zeros(nseq, nbatch, nptiles, device=self.device)
        outputs[-gsize:, :, :] = preds
        return outputs

    def __str__(self):
        """Provide a wee bit more info here."""
        return f"{self.LABEL}.{self.batch_pred}"
