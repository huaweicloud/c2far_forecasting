"""Special BatchPredictorMgr that knows how to handle multivariate
data and do batch prediction for multiple sub-series.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/multivariate/hf_multiv_batch_predictor_mgr.py
from collections import defaultdict
import torch
from sutranets.ml.batch_predictor import BatchPredictor
from sutranets.ml.batch_predictor_utils import BatchPredictorUtils
from sutranets.ml.bin_covariates import BinCovariates
from sutranets.ml.multivariate.multi_freq_mapper import MultiFreqMapper


class HFMultivBatchPredictorMgr():
    """Given an LSTM evaluator that has a multi-freq model and other key
    parameters, it converts a list of conditioning sequences into
    prediction quantiles for each sub-series, in a list.

    """
    def __init__(self, multi_lstm_evaluator,
                 coarse_cutoffs, coarse_low, coarse_high, *, nstride,
                 trivial_min_nonzero_frac, nsamples, device, confidence, bdecoder, nfine_bins,
                 nfine2_bins, extremas, sub_csize, mv_no_prevs, sub_gsize, sub_lf_period):
        self.nsamples = nsamples
        self.confidence = confidence
        self.device = device
        self.sub_csize = sub_csize
        self.sub_gsize = sub_gsize
        self.include_prevs = not mv_no_prevs
        self.lstm_evaluator = multi_lstm_evaluator
        sub_evals = list(multi_lstm_evaluator.yield_sub_evaluators())
        self.nsub_series = len(sub_evals)
        self.batch_predictors = []
        for i, sub_eval in enumerate(sub_evals):
            batchp = BatchPredictor(
                sub_eval, coarse_cutoffs, coarse_low, coarse_high,
                csize=sub_csize, nsize=sub_csize, nstride=sub_gsize, gsize=sub_gsize,
                trivial_min_nonzero_frac=trivial_min_nonzero_frac, nsamples=nsamples,
                device=device, confidence=confidence, bdecoder=bdecoder, nfine_bins=nfine_bins,
                nfine2_bins=nfine2_bins, extremas=extremas, lagfeat_period=sub_lf_period)
            self.batch_predictors.append(batchp)

    def __str__(self):
        mystr = f"HFMultivBP.nsub={self.nsub_series}."
        mystr += "---".join([str(bp) for bp in self.batch_predictors])
        return mystr

    def prep_inputs(self, originals):
        """Prepare the inputs - which come from originals, and which may span
        beyond csize - for the prediction call.

        """
        all_origs = originals.get_lst_of_tensors()
        prepared_origs = []
        for sub_origs in all_origs:
            sub_origs = BatchPredictorUtils.prep_batch_pred_inputs(sub_origs, self.sub_csize, self.sub_gsize)
            prepared_origs.append(sub_origs)
        return prepared_origs

    def prep_outputs(self, all_ptiles):
        """Prepare the outputs.  For multivariate outputs, we just keep the
        outputs in a list, one for each sub-series, where each element
        in the list is prepared in the standard batch_predictor way.

        """
        all_outputs = []
        for ptiles in all_ptiles:
            outs = BatchPredictorUtils.prep_batch_pred_outputs(ptiles, self.sub_csize, self.sub_gsize, self.device)
            all_outputs.append(outs)
        return all_outputs

    def __init_bin_covars(self, originals_lst):
        """Handle initializing the relevant bin covariates.  We take in all
        the cwindows (vals_lsts) across all the covariates, of size
        NBATCH, and we set up the BinCovariates so that (1) sequences
        of them are available for the creation of the initial batch,
        and (2) expanded copies of the *last* value in each vals_lsts
        is available as covariates for generation.

        """
        cwin_raw_lst = []
        gwin_raw_lst = []
        for vals_lst in originals_lst:
            vals_seq = vals_lst.transpose(0, 1)
            cwin_raw_lst.append(vals_seq)
            gwin_raws = vals_seq[-1].unsqueeze(0)
            gwin_raws = torch.repeat_interleave(gwin_raws, self.nsamples, dim=1)
            gwin_raw_lst.append(gwin_raws)
        cwin_bcs = BinCovariates(raw_lst=cwin_raw_lst, include_prevs=self.include_prevs)
        gwin_bcs = BinCovariates(raw_lst=gwin_raw_lst, include_prevs=self.include_prevs)
        return cwin_bcs, gwin_bcs

    def __call__(self, originals_lst):
        """Get the predictions for each set of vals in the given values lists,
        and output the percentiles for each.

        """
        do_hid_state = self.lstm_evaluator.get_do_init_hidden()
        self.lstm_evaluator.set_do_init_hidden(False)
        cwin_bcovars, gwin_bcovars = self.__init_bin_covars(originals_lst)
        batchp_gens = []
        for i, (vals_lst, batchp) in enumerate(zip(originals_lst, self.batch_predictors)):
            batchp_gen_i = batchp(vals_lst, cwindow_bin_covars=cwin_bcovars,
                                  gen_bin_covars=gwin_bcovars)
            batchp_gens.append(batchp_gen_i)
        all_qtiles_dct = defaultdict(list)
        with torch.no_grad():
            for j in range(self.sub_gsize):
                for i, batchp_gen_i in enumerate(batchp_gens):
                    cwin_bcovars.set_target_level(i), gwin_bcovars.set_target_level(i)
                    gen_ij = next(batchp_gen_i)
                    all_qtiles_dct[i].append(gen_ij)
                    gwin_bcovars.replace_slice(i, gen_ij.transpose(0, 1))
        all_qtiles = []
        for i in range(self.nsub_series):
            curr_gen = torch.cat(all_qtiles_dct[i], dim=1)
            qtiles = BatchPredictorUtils.make_ptiles(
                self.confidence, self.nsamples, self.sub_gsize, curr_gen)
            all_qtiles.append(qtiles)
        self.lstm_evaluator.set_do_init_hidden(do_hid_state)
        return all_qtiles

    @classmethod
    def create(cls, multi_lstm_eval, coarse_cutoffs, args,
               bdecoder):
        """Factory method to create BatchPredictorMgr from these arguments
        (including CLI-style dict arguments).

        """
        sub_gsize = MultiFreqMapper.divide_and_check(args.gsize, args.nsub_series)
        sub_lf_period = MultiFreqMapper.divide_and_check(args.lagfeat_period, args.nsub_series)
        bpm = cls(multi_lstm_eval, coarse_cutoffs,
                  args.coarse_low, args.coarse_high,
                  nstride=args.nstride, nsamples=args.nsamples,
                  trivial_min_nonzero_frac=args.trivial_min_nonzero_frac,
                  device=args.device, confidence=args.confidence_pct,
                  bdecoder=bdecoder, nfine_bins=args.nfine_bins,
                  nfine2_bins=args.nfine2_bins, extremas=args.extremas,
                  sub_csize=args.sub_csize, mv_no_prevs=args.mv_no_prevs, sub_gsize=sub_gsize,
                  sub_lf_period=sub_lf_period)
        return bpm
