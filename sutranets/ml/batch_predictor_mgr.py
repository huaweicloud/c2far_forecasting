"""Wrapper around BatchPredictor that takes in conditioning sequences,
gets samples for them from BatchPredictor, then computes the
appropriate quantiles.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/batch_predictor_mgr.py
import torch
from sutranets.ml.batch_predictor import BatchPredictor
from sutranets.ml.batch_predictor_utils import BatchPredictorUtils


class BatchPredictorMgr():
    """Input: list of conditioning sequences, output: prediction
    quantiles.

    """

    def __init__(self, lstm_evaluator, coarse_cutoffs, coarse_low, coarse_high,
                 *, csize, nsize, nstride, gsize, trivial_min_nonzero_frac, nsamples, device,
                 confidence, bdecoder, nfine_bins, nfine2_bins, extremas, lagfeat_period):
        self.nsamples = nsamples
        self.confidence = confidence
        self.csize = csize
        self.gsize = gsize
        self.device = device
        self.lstm_evaluator = lstm_evaluator
        self.bpredictor = BatchPredictor(
            lstm_evaluator, coarse_cutoffs, coarse_low, coarse_high,
            csize=csize, nsize=nsize, nstride=nstride, gsize=gsize,
            trivial_min_nonzero_frac=trivial_min_nonzero_frac, nsamples=nsamples,
            device=device, confidence=confidence, bdecoder=bdecoder, nfine_bins=nfine_bins,
            nfine2_bins=nfine2_bins, extremas=extremas, lagfeat_period=lagfeat_period)

    def __str__(self):
        """Provide some quick info about what kind of predictor this is."""
        return str(self.bpredictor)

    def prep_inputs(self, originals):
        """Prepare the inputs, which may span beyond csize, for the prediction
        call.

        """
        return BatchPredictorUtils.prep_batch_pred_inputs(originals, self.csize, self.gsize)

    def prep_outputs(self, ptiles):
        """Prepare the outputs.

        """
        return BatchPredictorUtils.prep_batch_pred_outputs(
            ptiles, self.csize, self.gsize, self.device)

    def __call__(self, vals_lst):
        """Get the predictions for each set of vals in the given values lists,
        and output the nsample samples for each.

        """
        do_hid_state = self.lstm_evaluator.get_do_init_hidden()
        self.lstm_evaluator.set_do_init_hidden(False)
        with torch.no_grad():
            slices = list(self.bpredictor(vals_lst))
        all_gen = torch.cat(slices, dim=1)
        qtiles = BatchPredictorUtils.make_ptiles(
            self.confidence, self.nsamples, self.gsize, all_gen)
        self.lstm_evaluator.set_do_init_hidden(do_hid_state)
        return qtiles

    @classmethod
    def create(cls, lstm_eval, coarse_cutoffs, args, bdecoder):
        """Factory method to take care of creating the BatchPredictorMgr from
        these arguments (including CLI-style dict arguments).

        """
        bpm = cls(lstm_eval, coarse_cutoffs,
                  args.coarse_low, args.coarse_high, csize=args.csize,
                  nsize=args.nsize, nstride=args.nstride,
                  gsize=args.gsize, trivial_min_nonzero_frac=args.trivial_min_nonzero_frac,
                  nsamples=args.nsamples, device=args.device, confidence=args.confidence_pct,
                  bdecoder=bdecoder, nfine_bins=args.nfine_bins,
                  nfine2_bins=args.nfine2_bins, extremas=args.extremas,
                  lagfeat_period=args.lagfeat_period)
        return bpm
