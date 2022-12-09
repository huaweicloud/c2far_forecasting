"""Use a trained *joint* LSTM model to generate sequences of the next
ngen values, priming on an existing set of conditioning of data (each
csize in length).

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/joint_generator.py
import logging
import torch
from c2far.dataprep.data_utils import DataUtils
from c2far.dataprep.digitize_utils import DigitizeUtils
from c2far.ml.constants import \
    ExampleKeys, INPUT_DTYPE, IntraBinDecoder, ProbError
from c2far.ml.fc_digitizer import FCDigitizer
from c2far.ml.bin_tensor_maker import BinTensorMaker
from c2far.ml.generator import Generator
from c2far.ml.loss_functions.joint_cross_entropy_loss import JointCrossEntropyLoss
logger = logging.getLogger("c2far.ml.joint_generator")
TOLERANCE = 1e-6


class JointGenerator(Generator):
    """Class to generate a sequence of values given a trained JointLSTM
    model.  Similar to generator (see there for further comments).

    """
    def __init__(self, ncoarse, nfine, device, evaluator,
                 coarse_cutoffs, coarse_low, coarse_high, nsize, ngen,
                 bdecoder, *, extremas):
        """Initialize with the JOINT neural net evaluator and the other things
        we need for generation.

        Arguments:

        ncoarse: Int, the number of coarse bins.

        nfine: Int, the number of fine bins.

        device: String, e.g. "cuda:0" or "cpu"

        evaluator: JointLSTMEvaluator: what we use to call
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

        Keyword-only Arguments:

        extremas: Boolean, if True, use the net's fully-connected
        sub-network to generate outputs when in the extreme bins, else
        False.

        """
        self.ncoarse = ncoarse
        self.nfine = nfine
        self.device = device
        self.evaluator = evaluator
        self.nsize = nsize
        coarse_cutoffs = coarse_cutoffs.to(device)
        # Store these which help us find our pos in each coarse bin:
        self.expanded_cuts = FCDigitizer.expand_cuts(coarse_cutoffs, coarse_low, coarse_high)
        # To unmap fines, we need the fine cutoffs (on the device):
        fcutoffs = FCDigitizer.get_fine_cutoffs(self.nfine, device=device)
        low_high_fcutoffs = DigitizeUtils.torch_prepare_cutoffs(
            fcutoffs, fixed_lower=0.0, fixed_upper=1.0)
        self.lower_fcutoffs, self.upper_fcutoffs = low_high_fcutoffs
        self.ngen = ngen
        self.extremas = extremas
        if bdecoder not in [IntraBinDecoder.UNIFORM]:
            raise RuntimeError(f"Unsupported bdecoder {bdecoder}")
        self.bdecoder = bdecoder
        if self.extremas:
            _, self.pareto_lo, self.pareto_hi = JointCrossEntropyLoss.process_fine_cutoffs(coarse_cutoffs, nfine)
        else:
            self.pareto_lo = self.pareto_hi = None
        self.softplus = torch.nn.Softplus()

    def __replace_extremas(self, decoded, binned_coarses, binned_fines, extrema_inputs):
        """Find the extrema points, determine samples for them, and then
        replace the parts of decoded corresponding to them.

        Arguments:

        decoded: Tensor[Float]: NBATCH, the normalized decoded values.

        binned_coarses: Tensor: NBATCH, the bin idxs for the coarses.

        binned_fines: Tensor: NBATCH, the bin idxs for the fines.

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
        hi_mask = torch.logical_and(binned_coarses == self.ncoarse - 1, binned_fines == self.nfine - 1)
        decoded = self._sample_and_replace_extremas(decoded, extrema_inputs, lo_mask, hi_mask)
        return decoded

    def _decode_batch(self, bcoarses, bfines, extrema_inputs, nmin, nmax):
        """Turn the batches of binned coarse/fine values into real output
        values.

        Arguments:

        bcoarses: Tensor: NBATCH x 1, the coarse idxs to decode

        bfines: Tensor: NBATCH x 1, the fine idxs to decode

        extrema_inputs: 1 x NBATCH x NINPUTS=1, the one input per batch
        for the extrema forward function.

        nmin/nmax: each NBATCH x 1

        Returns:

        Tensor: NBATCH, the decoded, but normed values, or None if not doing extremas.

        Tensor: NBATCH, the decoded/unnormed values.

        """
        bcoarses, bfines = bcoarses.reshape(-1), bfines.reshape(-1)
        coarse_bins_lower = self.expanded_cuts[bcoarses]
        coarse_bins_higher = self.expanded_cuts[bcoarses + 1]
        # Now, decode the fine bins to points UNIFORMLY between 0 and
        # 1 - except the first, which goes to 0.0 via special_value_decode:
        dfines = DigitizeUtils.torch_decode_values(
            bfines, self.lower_fcutoffs, self.upper_fcutoffs)
        decoded = (dfines * (coarse_bins_higher - coarse_bins_lower) +
                   coarse_bins_lower)
        if self.extremas:
            decoded = self.__replace_extremas(decoded, bcoarses, bfines, extrema_inputs)
        decoded = decoded.reshape(-1, 1)
        unnormed = self._unnormalize(decoded, nmin, nmax)
        if self.extremas:
            return decoded, unnormed
        return None, unnormed

    def _get_binned(self, coarse_inputs, fine_inputs):
        """Helper to get the coarse/fine binned values and ccoarse, moved to
        this separate function purely to reduce complexity of
        build_outseqs.

        Arguments:

        coarse_inputs: Tensor[1 x NBATCH x NSUBSET], the subset of
        coarse features, for each element of the batch, for the latest
        element of the sequence.

        fine_inputs: Tensor[1 x NBATCH x NSUBSET], the subset of
        fine features, for each element of the batch, for the latest
        element of the sequence.

        Returns:

        binned_coarses: Tensor: NBATCH x 1, the coarse idxs

        binned_fines: Tensor: NBATCH x 1, the fine idxs to

        ccoarses: Tensor[NBATCH x NCOARSE]: 1-hot-encoded features for "current" coarse bins.

        """
        binned_coarses = super()._get_binned(coarse_inputs)
        ccoarses = torch.nn.functional.one_hot(
            binned_coarses.reshape(-1), num_classes=self.ncoarse).to(INPUT_DTYPE)
        fine_inputs = BinTensorMaker.replace_ccoarse(fine_inputs, ccoarses)
        fine_output = self.evaluator.batch_forward_fine(fine_inputs)
        fine_logits = fine_output[-1]
        fine_probs = torch.softmax(fine_logits, dim=1)
        try:
            binned_fines = torch.multinomial(fine_probs, 1)
        except RuntimeError as multi_err:
            raise ProbError("Error with torch.multinomial") from multi_err
        return binned_coarses, binned_fines, ccoarses

    def _build_outseqs(self, last_inputs, originals):
        """Starting with the last set of inputs from the conditioning window
        (and a network that is already conditioned), auto-regressively
        call the coarse and fine sub-networks in order to build output
        sequences for all sequences in the batch.

        Arguments:

        last_inputs: 1 x NBATCH x NFEATS, final inputs, with dummy
        ccoarse.

        originals: NSEQ+2 x NBATCH x 1, needed for getting nmin, nmax.

        -- In all cases, NSEQ is length of conditioning window.

        Returns:

        outseqs, torch.tensor(Float), of shape NGEN x NBATCH x 1,
        where NBATCH is perhaps NERIES*NSAMPLES if from batch_predictor

        """
        csize = len(originals) - 2
        nmin, nmax = DataUtils.get_nmin_nmax(csize, self.nsize,
                                             originals)
        all_gen = []
        for _ in range(self.ngen):
            coarse_inputs, fine_inputs, extrema_inputs = BinTensorMaker.extract_coarse_fine(
                last_inputs, self.ncoarse, extremas=self.extremas)
            binned_coarses, binned_fines, ccoarses = self._get_binned(coarse_inputs, fine_inputs)
            normed, generated = self._decode_batch(
                binned_coarses, binned_fines, extrema_inputs, nmin, nmax)
            all_gen.append(generated)
            pfine = torch.nn.functional.one_hot(
                binned_fines.reshape(-1),
                num_classes=self.nfine).to(INPUT_DTYPE)
            pcoarse = ccoarses
            # Loop invariant: only valid part of last_inputs for next
            # iter is ccoarse_inputs part: So replace non-ccoarse here
            # and then ccoarse parts of last_inputs in loop body:
            last_inputs = BinTensorMaker.reset_ccoarse_inputs(
                last_inputs, pcoarse, pfine=pfine, normed=normed)
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
            # -- These are also NGEN x NBATCH x 1
            self.evaluator.set_do_init_hidden(do_hid_state)
            # These are also NSERIESxNSAMPLES x NGEN:
            return outseqs
