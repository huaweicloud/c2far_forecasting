"""Triple generator subclass of generator, for coarse/fine/fine2 binnings.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/triple_generator.py
import logging
import torch
from sutranets.dataprep.data_utils import DataUtils
from sutranets.dataprep.digitize_utils import DigitizeUtils
from sutranets.ml.constants import INPUT_DTYPE, ProbError
from sutranets.ml.fc_digitizer import FCDigitizer
from sutranets.ml.bin_tensor_maker import BinTensorMaker
from sutranets.ml.joint_generator import JointGenerator
from sutranets.ml.loss_functions.triple_cross_entropy_loss import TripleCrossEntropyLoss
logger = logging.getLogger("sutranets.ml.triple_generator")
TOLERANCE = 1e-6


class TripleGenerator(JointGenerator):
    """Class to generate a sequence of values given a trained TripleLSTM
    model.

    """
    def __init__(self, ncoarse, nfine, nfine2, device, evaluator,
                 coarse_cutoffs, coarse_low,
                 coarse_high, nsize, ngen, bdecoder, *,
                 extremas, lagfeat_period):
        self.device = self.evaluator = self.nsize = None
        self.coarse_cutoffs = None
        self.lagfeat_period = self.ngen = self.extremas = self.bdecoder = self.softplus = None
        self.ncoarse = self.nfine = self.coarse_cuts_full = self.fcutoffs = None
        self._init_standard_vars(
            device, evaluator, nsize, coarse_cutoffs,
            lagfeat_period, ngen, extremas, bdecoder,
            ncoarse, nfine, coarse_low, coarse_high)
        self.nfine2 = nfine2
        if self.extremas:
            _, self.pareto_lo, self.pareto_hi = TripleCrossEntropyLoss.process_fine2_cutoffs(
                coarse_cutoffs, nfine, nfine2)
        else:
            self.pareto_lo = self.pareto_hi = None
        self.fine_cuts_full = FCDigitizer.expand_cuts(self.fcutoffs, 0, 1)
        f2cutoffs = FCDigitizer.get_fine_cutoffs(self.nfine2, device=device)
        low_high_f2cutoffs = DigitizeUtils.torch_prepare_cutoffs(
            f2cutoffs, fixed_lower=0.0, fixed_upper=1.0)
        self.lower_f2cutoffs, self.upper_f2cutoffs = low_high_f2cutoffs

    def __replace_extremas(self, decoded, binned_coarses, binned_fines,
                           binned_fines2, extrema_inputs):
        """Find the extrema points, determine samples for them, and then
        replace the parts of decoded corresponding to them.

        """
        if not self.extremas:
            return decoded
        lo_mask = torch.logical_and(binned_coarses == 0, binned_fines == 0)
        lo_mask = torch.logical_and(lo_mask, binned_fines2 == 0)
        hi_mask = torch.logical_and(binned_coarses == self.ncoarse - 1, binned_fines == self.nfine - 1)
        hi_mask = torch.logical_and(hi_mask, binned_fines2 == self.nfine2 - 1)
        decoded = self._sample_and_replace_extremas(decoded, extrema_inputs, lo_mask, hi_mask)
        return decoded

    def _decode_batch(self, bcoarses, bfines, bfines2, extrema_inputs,
                      nmin, nmax):
        """Turn the batches of binned coarse/fine/fine2 idxs into output
        values.

        """
        bcoarses, bfines, bfines2 = bcoarses.reshape(-1), bfines.reshape(-1), bfines2.reshape(-1)
        coarse_b_low = self.coarse_cuts_full[bcoarses]
        coarse_b_hi = self.coarse_cuts_full[bcoarses + 1]
        fine_b_low = self.fine_cuts_full[bfines]
        fine_b_hi = self.fine_cuts_full[bfines + 1]
        dfines2 = DigitizeUtils.torch_decode_values(
            bfines2, self.lower_f2cutoffs, self.upper_f2cutoffs)
        lower_fines = fine_b_low * (coarse_b_hi - coarse_b_low) + coarse_b_low
        upper_fines = fine_b_hi * (coarse_b_hi - coarse_b_low) + coarse_b_low
        decoded = dfines2 * (upper_fines - lower_fines) + lower_fines
        decoded = self.__replace_extremas(decoded, bcoarses, bfines, bfines2, extrema_inputs)
        decoded = decoded.reshape(-1, 1)
        unnormed = self._unnormalize(decoded, nmin, nmax)
        if self.extremas:
            return decoded, unnormed
        return None, unnormed

    def _get_binned(self, coarse_inputs, fine_inputs, fine2_inputs):
        """Get the coarse/fine/fine2 binned values and ccoarse/cfine.

        """
        binned_coarses, binned_fines, ccoarses, cfines = super()._get_binned(
            coarse_inputs, fine_inputs)
        fine2_inputs = BinTensorMaker.replace_cfine(fine2_inputs, cfines)
        fine2_output = self.evaluator.batch_forward_fine2(fine2_inputs)
        fine2_logits = fine2_output[-1]
        fine2_probs = torch.softmax(fine2_logits, dim=1)
        try:
            binned_fines2 = torch.multinomial(fine2_probs, 1)
        except RuntimeError as multi_err:
            raise ProbError("Error with torch.multinomial") from multi_err
        cfines2 = torch.nn.functional.one_hot(
            binned_fines2.reshape(-1), num_classes=self.nfine2).to(INPUT_DTYPE)
        return binned_coarses, binned_fines, binned_fines2, ccoarses, cfines, cfines2

    def _build_outseqs(self, last_inputs, originals, cwin_coarses,
                       cwin_fines, cwin_fines2, bin_covars):
        """Starting with the given set of coarse, as well as the final
        fine_inputs, auto-regressively call the coarse, fine, and
        fine2 sub-networks in order to build output sequences for all
        sequences in the batch.

        """
        csize = len(originals) - 2
        nmin, nmax = DataUtils.get_nmin_nmax(csize, self.nsize,
                                             originals, check_trivial=False)
        last_inputs = self._init_covar_inputs(last_inputs, bin_covars, self.ncoarse, self.nfine, self.nfine2)
        lagp_coarses, lagp_fines, lagp_fines2 = self._init_lagperiod_arrays(
            cwin_coarses, cwin_fines, cwin_fines2)
        for idx in range(self.ngen):
            coarse_inputs, fine_inputs, fine2_inputs, extrema_inputs = BinTensorMaker.extract_coarse_fine_fine2(
                last_inputs, self.ncoarse, self.nfine, extremas=self.extremas)
            binned_coarses, binned_fines, binned_fines2, ccoarses, cfines, cfines2 = self._get_binned(
                coarse_inputs, fine_inputs, fine2_inputs)
            lagp_coarses, lagp_fines, lagp_fines2 = self._update_lagperiod_arrays(
                lagp_coarses, lagp_fines, lagp_fines2, binned_coarses, binned_fines, binned_fines2)
            normed, generated = self._decode_batch(
                binned_coarses, binned_fines, binned_fines2, extrema_inputs, nmin, nmax)
            yield generated
            if idx == self.ngen - 1:
                break
            lcoarse, lfine, lfine2 = self._get_lagfeats(
                lagp_coarses, lagp_fines, lagp_fines2, self.ncoarse, self.nfine, self.nfine2)
            covar_coarse, covar_fine, covar_fine2 = self._get_covar_feats(
                bin_covars, idx + 1, self.ncoarse, self.nfine, self.nfine2)
            pcoarse, pfine, pfine2 = ccoarses, cfines, cfines2
            last_inputs = BinTensorMaker.reset_ccoarse_inputs(
                last_inputs, pcoarse=pcoarse, pfine=pfine,
                pfine2=pfine2, normed=normed,
                lcoarse=lcoarse, lfine=lfine, lfine2=lfine2,
                covar_coarse=covar_coarse, covar_fine=covar_fine, covar_fine2=covar_fine2)

    def __call__(self, batch, bin_covars=None):
        """Generate self.ngen points forward on the given batch of data.

        """
        if batch is None:
            for _ in range(self.ngen):
                yield None
        inputs, originals, cwin_coarses, cwin_fines, cwin_fines2 = self._get_cwindow_seqs(
            batch, do_fines=True, do_fines2=True)
        batch_size = inputs.shape[1]
        self.evaluator.run_init_hidden(batch_size)
        _, _, originals, cwin_coarses, cwin_fines, cwin_fines2 = self.evaluator.batch_forward_outputs(
            inputs[:-1], None, originals, cwin_coarses, cwin_fines, cwin_fines2)
        last_inputs = inputs[-1:]
        for generated in self._build_outseqs(
                last_inputs, originals, cwin_coarses, cwin_fines, cwin_fines2, bin_covars):
            yield generated
