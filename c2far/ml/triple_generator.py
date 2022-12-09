"""Use a trained *triple* LSTM model to generate sequences of the next
ngen values, priming on an existing set of conditioning of data (each
csize in length).

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/triple_generator.py

import logging
import torch
from c2far.dataprep.data_utils import DataUtils
from c2far.dataprep.digitize_utils import DigitizeUtils
from c2far.ml.constants import \
    ExampleKeys, INPUT_DTYPE, IntraBinDecoder, ProbError
from c2far.ml.fc_digitizer import FCDigitizer
from c2far.ml.bin_tensor_maker import BinTensorMaker
from c2far.ml.joint_generator import JointGenerator
from c2far.ml.loss_functions.triple_cross_entropy_loss import TripleCrossEntropyLoss
logger = logging.getLogger("c2far.ml.triple_generator")
TOLERANCE = 1e-6


class TripleGenerator(JointGenerator):
    """Class to generate a sequence of values given a trained TripleLSTM
    model.  Similar to generator (see there for further comments).

    """
    def __init__(self, ncoarse, nfine, nfine2, device, evaluator,
                 coarse_cutoffs, coarse_low, coarse_high, nsize, ngen,
                 bdecoder, *, extremas):
        """Initialize with the triple neural net evaluator and the other
        things we need for generation.

        Arguments:

        ncoarse: Int, the number of coarse bins.

        nfine: Int, the number of fine bins.

        nfine2: Int, the number of fine2 bins.

        device: String, e.g. "cuda:0" or "cpu"

        evaluator: TripleLSTMEvaluator: what we use to call
        batch_forward_coarse(), batch_forward_fine(), etc.

        coarse_cutoffs: Tensor[NBINS], for encoding coarse features and
        converting vals to origs

        coarse_low: Float, the lowest implicit boundary for the coarse
        cutoffs.

        coarse_high: Float, the highest implicit boundary for the
        coarse cutoffs.

        nsize: Int, length of normalization window (last part of
        conditioning window)

        ngen: Int, length of generation window

        bdecoder: IntraBinDecoder, the method for decoding the value
        within each coarse/fine bin pair.

        Keyword-only arguments:

        extremas: Boolean, if True, use the net's fully-connected
        sub-network to generate outputs when in the extreme bins.

        """
        if bdecoder not in [IntraBinDecoder.UNIFORM]:
            raise RuntimeError(f"Unsupported bdecoder {bdecoder}")
        self.bdecoder = bdecoder
        self.ncoarse = ncoarse
        self.nfine = nfine
        self.nfine2 = nfine2
        self.device = device
        self.evaluator = evaluator
        self.nsize = nsize
        self.ngen = ngen
        self.extremas = extremas
        coarse_cutoffs = coarse_cutoffs.to(device)
        # Store these which help us find our pos in each coarse bin:
        self.coarse_cuts_full = FCDigitizer.expand_cuts(coarse_cutoffs, coarse_low, coarse_high)
        # To unmap fines, we need the fine cutoffs (on the device):
        fcutoffs = FCDigitizer.get_fine_cutoffs(self.nfine, device=device)
        # Expanded fines for finding our fine proportions, for getting fine2:
        self.fine_cuts_full = FCDigitizer.expand_cuts(fcutoffs, 0, 1)
        # To unmap fines2, we need the fine2 cutoffs (on the device):
        f2cutoffs = FCDigitizer.get_fine_cutoffs(self.nfine2, device=device)
        low_high_f2cutoffs = DigitizeUtils.torch_prepare_cutoffs(
            f2cutoffs, fixed_lower=0.0, fixed_upper=1.0)
        self.lower_f2cutoffs, self.upper_f2cutoffs = low_high_f2cutoffs
        if self.extremas:
            _, self.pareto_lo, self.pareto_hi = TripleCrossEntropyLoss.process_fine2_cutoffs(
                coarse_cutoffs, nfine, nfine2)
        else:
            self.pareto_lo = self.pareto_hi = None
        self.softplus = torch.nn.Softplus()

    def __replace_extremas(self, decoded, binned_coarses, binned_fines,
                           binned_fines2, extrema_inputs):
        """Find the extrema points, determine samples for them, and then
        replace the parts of decoded corresponding to them.

        Arguments:

        decoded: Tensor[Float]: NBATCH, the normalized decoded values.

        binned_coarses: Tensor: NBATCH, the bin idxs for the coarses.

        binned_fines: Tensor: NBATCH, the bin idxs for the fines.

        binned_fines2: Tensor: NBATCH, the fine2 idxs to decode

        extrema_inputs: 1 x NBATCH x NINPUTS=1, the one input per batch
        for the extrema forward function.

        Returns:

        decoded: Tensor[Float]: NBATCH x 1, the normalized decoded
        values, but with values in extrema bins replaced by Pareto
        samples.

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
        """Turn the batches of binned coarse/fine/fine2 idxs into real output
        values.

        Arguments:

        bcoarses: Tensor: NBATCH x 1, the coarse idxs to decode

        bfines: Tensor: NBATCH x 1, the fine idxs to decode

        bfines2: Tensor: NBATCH x 1, the fine2 idxs to decode

        extrema_inputs: 1 x NBATCH x NINPUTS=1, the one input per batch
        for the extrema forward function.

        nmin/nmax: each NBATCH x 1

        Returns:

        Tensor: NBATCH, the decoded, but normed values, or None if not doing extremas.

        Tensor: NBATCH, the decoded/unnormed values.

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
        if self.extremas:
            decoded = self.__replace_extremas(decoded, bcoarses, bfines, bfines2, extrema_inputs)
        decoded = decoded.reshape(-1, 1)
        unnormed = self._unnormalize(decoded, nmin, nmax)
        if self.extremas:
            return decoded, unnormed
        return None, unnormed

    def _get_binned(self, coarse_inputs, fine_inputs, fine2_inputs):
        """Helper to get the coarse/fine/fine2 binned values and
        ccoarse/cfine, moved to this separate function purely to
        reduce complexity of build_outseqs.

        Arguments:

        coarse_inputs: Tensor[1 x NBATCH x NSUBSET], the subset of
        coarse features, for each element of the batch, for the latest
        element of the sequence.

        fine_inputs: Tensor[1 x NBATCH x NSUBSET], the subset of
        fine features, for each element of the batch, for the latest
        element of the sequence.

        fine2_inputs: Tensor[1 x NBATCH x NSUBSET], the subset of
        fine2 features, for each element of the batch, for the latest
        element of the sequence.

        Returns:

        binned_coarses: Tensor: NBATCH x 1, the coarse idxs

        binned_fines: Tensor: NBATCH x 1, the fine idxs to

        binned_fines2: Tensor: NBATCH x 1, the fine2 idxs to

        ccoarses: Tensor[NBATCH x NCOARSE]: 1-hot-encoded features for "current" coarse bins.

        cfines: Tensor[NBATCH x NFINE]: 1-hot-encoded features for "current" fine bins.

        """
        binned_coarses, binned_fines, ccoarses = super()._get_binned(coarse_inputs, fine_inputs)
        cfines = torch.nn.functional.one_hot(
            binned_fines.reshape(-1), num_classes=self.nfine).to(INPUT_DTYPE)
        fine2_inputs = BinTensorMaker.replace_cfine(fine2_inputs, cfines)
        fine2_output = self.evaluator.batch_forward_fine2(fine2_inputs)
        fine2_logits = fine2_output[-1]
        fine2_probs = torch.softmax(fine2_logits, dim=1)
        try:
            binned_fines2 = torch.multinomial(fine2_probs, 1)
        except RuntimeError as multi_err:
            raise ProbError("Error with torch.multinomial") from multi_err
        return binned_coarses, binned_fines, binned_fines2, ccoarses, cfines

    def _build_outseqs(self, last_inputs, originals):
        """Starting with the given set of *coarse* logits, as well as the
        final fine_inputs, auto-regressively call the coarse and fine
        sub-networks in order to build output sequences for all
        sequences in the batch.

        Arguments:

        last_inputs: 1 x NBATCH x NFEATS, final inputs, with dummy
        ccoarse.

        originals: NSEQ+2 x NBATCH x 1, needed for getting nmin, nmax.

        -- In all cases, NSEQ is length of conditioning window.

        Returns:

        outseqs, torch.tensor(Float), of shape NGEN x NBATCH x 1,
        where NBATCH is perhaps NERIES*NSAMPLES if from batch_predictor

        """
        # n.b., this is all vectorized over batches:
        csize = len(originals) - 2
        nmin, nmax = DataUtils.get_nmin_nmax(csize, self.nsize,
                                             originals)
        all_gen = []
        for _ in range(self.ngen):
            coarse_inputs, fine_inputs, fine2_inputs, extrema_inputs = BinTensorMaker.extract_coarse_fine_fine2(
                last_inputs, self.ncoarse, self.nfine, extremas=self.extremas)
            binned_coarses, binned_fines, binned_fines2, ccoarses, cfines = self._get_binned(
                coarse_inputs, fine_inputs, fine2_inputs)
            normed, generated = self._decode_batch(
                binned_coarses, binned_fines, binned_fines2, extrema_inputs, nmin, nmax)
            all_gen.append(generated)
            pfine2 = torch.nn.functional.one_hot(
                binned_fines2.reshape(-1),
                num_classes=self.nfine2).to(INPUT_DTYPE)
            pcoarse = ccoarses
            pfine = cfines
            # Loop invariant: only valid part of last_inputs for next
            # iter is ccoarse_inputs part: So replace non-ccoarse here
            # and then ccoarse parts of last_inputs in loop body:
            last_inputs = BinTensorMaker.reset_ccoarse_inputs(
                last_inputs, pcoarse=pcoarse, pfine=pfine,
                pfine2=pfine2, normed=normed)
        outseqs = torch.cat(all_gen, dim=1)
        return outseqs

    def __call__(self, batch):
        """Generate self.ngen points forward on the given batch of data.

        Arguments:

        batch: Dict[ExampleKeys] -> Tensors[Float], where tensors are
        in shape of Tensor[NSEQ x NBATCH=NSERIES*NSAMPLES x {}] (last
        dim depending on key).

        Returns:

        outseqs, torch.tensor(Float), of shape NBATCH x NGEN (just 2
        dim), all the samples as a batch.

        """
        inputs = batch[ExampleKeys.INPUT]
        originals = batch[ExampleKeys.ORIGINALS]
        batch_size = inputs.shape[1]
        self.evaluator.run_init_hidden(batch_size)
        with torch.no_grad():
            do_hid_state = self.evaluator.get_do_init_hidden()
            self.evaluator.set_do_init_hidden(False)
            _, _, originals = self.evaluator.batch_forward_outputs(
                inputs[:-1], None, originals)
            last_inputs = inputs[-1:]
            outseqs = self._build_outseqs(last_inputs, originals)
            self.evaluator.set_do_init_hidden(do_hid_state)
            return outseqs
