"""Standalone, self-contained predictor that can take in a BATCH of
conditioning sequences and generate the full probabilistic predictions
for each of them for the future.

Uses our 'Generator', 'JointGenerator', etc. internally.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/batch_predictor.py

import logging
import torch
from c2far.dataprep.data_utils import DataUtils
from c2far.ml.bin_tensor_maker import BinTensorMaker
from c2far.ml.collate_utils import CollateUtils
from c2far.ml.constants import ExampleKeys, ConfidenceFlags
from c2far.ml.continuous.cont_tensor_maker import ContTensorMaker
from c2far.ml.continuous.cont_generator import ContGenerator
from c2far.ml.fc_digitizer import FCDigitizer
from c2far.ml.generator import Generator
from c2far.ml.joint_generator import JointGenerator
from c2far.ml.triple_generator import TripleGenerator

logger = logging.getLogger("c2far.ml.batch_predictor")


class BatchPredictor():
    """Init with an LSTM evaluator, the coarse maps, and other key
    parameters.  Then, when you give a list of conditioning sequences,
    it gives predictions.

    """

    def __init__(self, lstm_evaluator, coarse_cutoffs, coarse_low,
                 coarse_high, *, csize, nsize, nstride, gsize, nsamples, device,
                 confidence, bdecoder, nfine_bins, nfine2_bins, extremas):
        """Set up for future prediction.  If coarse_cutoffs and nfine_bins are
        None, use the continuous generator.

        Positional arguments:

        lstm_evaluator: LSTMEvaluator: system passed to generator to
        generate the outputs auto-regressively.

        coarse_cutoffs: Tensor[NBINS], for encoding coarse features
        and converting coarses to origs, or None if doing continuous.

        coarse_low: Float, the lowest implicit boundary for the coarse
        cutoffs (if using).

        coarse_high: Float, the highest implicit boundary for the
        coarse cutoffs (if using).

        Keyword-only arguments:

        csize: Int, the length of the conditioning window.

        nsize: Int, length of normalization window (last part of
        conditioning window, nsize <= csize).

        nstride: Int, how far to gen on each pass, nstride <= gsize.

        gsize: Int, how far to generate into future.

        nsamples: Int, number of rollouts to do per series in order to
        get confidence intervals.

        device: String, e.g. "cuda:0" or "cpu"

        confidence: Int, a % between 0 and 100, indicating the *width*
        of the upper/lower confidence bands, or a ConfidenceFlags
        value as a special flag, e.g., to return all values between
        0.1 and 0.9 by 0.1 (for wQL).

        bdecoder: IntraBinDecoder, method for decoding values within
        each coarse bin.

        nfine_bins: Int, num. fine bins to use, or None if not using.

        nfine2_bins: Int, num. finer fine2 bins to use, or None if not using.

        extremas: Boolean, if True, use the net's fully-connected
        sub-network to generate outputs when in the extreme bins, else
        False.

        """
        self.nfine = nfine_bins
        self.nfine2 = nfine2_bins
        self.nsize = nsize
        self.nstride = nstride
        self.csize = csize
        self.gsize = gsize
        self.device = device
        self.nsamples = nsamples
        self.confidence = confidence
        self.bdecoder = bdecoder
        if coarse_cutoffs is not None:
            self.ncoarse = len(coarse_cutoffs) + 1
            self.tmaker, self.digitizer, self.gen = self.__make_binned_objects(
                lstm_evaluator, coarse_cutoffs, coarse_low, coarse_high,
                extremas)
        else:
            self.ncoarse = None
            self.tmaker, self.digitizer, self.gen = self.__make_cont_objects(lstm_evaluator)

    def __make_binned_objects(self, lstm_evaluator, coarse_cutoffs,
                              coarse_low, coarse_high, extremas):
        """Helper to handle making the binned tmaker/digitizer/generators."""
        coarse_cutoffs = coarse_cutoffs.to(self.device)
        tmaker = BinTensorMaker(coarse_cutoffs, coarse_low, coarse_high, nfine_bins=self.nfine,
                                nfine2_bins=self.nfine2, genmode=True, extremas=extremas)
        digitizer = FCDigitizer(coarse_cutoffs, coarse_low, coarse_high,
                                nfine_bins=self.nfine, nfine2_bins=self.nfine2, extremas=extremas)
        if self.nfine is None:
            gen = Generator(self.device, lstm_evaluator,
                            coarse_cutoffs, coarse_low, coarse_high, self.nsize, ngen=self.nstride,
                            bdecoder=self.bdecoder, extremas=extremas)
        elif self.nfine2 is None:
            gen = JointGenerator(self.ncoarse, self.nfine, self.device, lstm_evaluator,
                                 coarse_cutoffs, coarse_low, coarse_high, self.nsize,
                                 ngen=self.nstride, bdecoder=self.bdecoder, extremas=extremas)
        else:
            gen = TripleGenerator(self.ncoarse, self.nfine, self.nfine2, self.device, lstm_evaluator,
                                  coarse_cutoffs, coarse_low, coarse_high, self.nsize,
                                  ngen=self.nstride, bdecoder=self.bdecoder, extremas=extremas)
        return tmaker, digitizer, gen

    def __make_cont_objects(self, lstm_evaluator):
        """Helper to handle making the continuous tmaker/digitizer/generators."""
        tmaker = ContTensorMaker(genmode=True)
        # No need for a digitizer in this case:
        digitizer = None
        gen = ContGenerator(self.device, lstm_evaluator, self.nsize, ngen=self.nstride)
        return tmaker, digitizer, gen

    def __str__(self):
        """Provide info about what kind of predictor this is."""
        if self.ncoarse is None:
            mystr = "ContPred"
        else:
            if self.nfine is None:
                mystr = f"CoarsePred.{self.ncoarse}"
            elif self.nfine2 is None:
                mystr = f"JointPred.c{self.ncoarse}.f{self.nfine}"
            else:
                mystr = f"TriplePred.c{self.ncoarse}.f{self.nfine}.f2{self.nfine2}"
            # Binning ones should identify their bdecoder:
            mystr += f".{self.bdecoder.value}"
        mystr += f".n{self.nsamples}.conf{self.confidence}"
        return mystr

    def get_csize_gsize(self):
        """Provide mechanism for our clients to get csize/gsize from us.
        Note, clients don't need nsize, because this method is used to
        prune time series down to key parts, not to normalize (where
        nsize is needed).

        """
        return self.csize, self.gsize

    def __digitize(self, origs):
        """Digitize values into our usual coarse encoding.  Gets a
        tensor of origs, returns tensors for coarses.

        Arguments:

        origs: Tensor: NSEQ+2, the conditioning window for the series
        (plus the usual two extra values).  Note it is NOT a batch at
        this point.

        Returns

        encoded_coarses, encoded_fines,
        encoded_fines2, normed: Tensor[NSEQ+2], Tensor[NSEQ+1],
        Tensor[NSEQ+1], Tensor[NSEQ+1]

        """
        coarses, fines, fines2, normed = self.digitizer(origs, self.csize, self.nsize)
        return coarses, fines, fines2, normed

    def __make_one_binned_batch(self, vals, nextended):
        """Given the values for one time series, turn the data into a batch,
        for BINNED prediction.

        Arguments:

        vals: Tensor[Float]: NSEQ, tensor of values for one series.

        nextended: Int, how much to expand this batch (typically equal
        to self.nsamples [original set of values] or 1 [striding]) -
        basically how much to copy one ts in batch dimension.

        Returns:

        batch, Dict[ExampleKeys] -> Tensors[Float]: a batch of
        tensors, each tensor of shape Tensor[NSEQ x nextended x {}]
        (i.e., depends on value of nextended).

        """
        origs = vals  # Our usual name for vals in a batch
        coarses, fines, fines2, normed = self.__digitize(origs)
        inputs = self.tmaker.encode_input(coarses, fines=fines,
                                          fines2=fines2, normed=normed)
        nfeats = inputs.shape[2]
        coarses = coarses.reshape(-1, 1, 1)
        if fines is not None:
            fines = fines.reshape(-1, 1, 1)
        if fines2 is not None:
            fines2 = fines2.reshape(-1, 1, 1)
        origs = origs.reshape(-1, 1, 1)
        if nextended > 1:
            # expand is like repeat but with references (no memory cp):
            inputs = inputs.expand(-1, nextended, nfeats)
            # Likewise, let's expand the originals and coarses (which,
            # note, have one more element than our genmode inputs):
            origs = origs.expand(-1, nextended, 1)
            coarses = coarses.expand(-1, nextended, 1)
            if fines is not None:
                fines = fines.expand(-1, nextended, 1)
            if fines2 is not None:
                fines2 = fines2.expand(-1, nextended, 1)
        batch = {ExampleKeys.INPUT: inputs, ExampleKeys.ORIGINALS:
                 origs, ExampleKeys.COARSES: coarses}
        return batch

    def __make_one_cont_batch(self, vals, nextended):
        """Given the values for one time series, turn the data into a batch,
        for continous prediction.

        Arguments:

        vals: Tensor[Float]: NSEQ, tensor of values for one series.

        nextended: Int, how much to expand this batch (typically equal
        to self.nsamples [original set of values] or 1 [striding]) -
        basically how much to copy one ts in batch dimension.

        Returns:

        batch, Dict[ExampleKeys] -> Tensors[Float]: a batch of
        tensors, each tensor of shape Tensor[NSEQ x nextended x {}]
        (i.e., depends on value of nextended).

        """
        normed = DataUtils.torch_norm(self.csize, self.nsize, vals)
        inputs = self.tmaker.encode_input(normed)
        origs = vals.reshape(-1, 1, 1)
        if nextended > 1:
            inputs = inputs.expand(-1, nextended, 1)
            origs = origs.expand(-1, nextended, 1)
        batch = {ExampleKeys.INPUT: inputs, ExampleKeys.ORIGINALS: origs}
        return batch

    def __make_one_ts_batch(self, vals, nextended):
        """Given the values for one time series, turn the data into a batch.

        Arguments:

        vals: Tensor[Float]: NSEQ, tensor of values for one series.

        nextended: Int, how much to expand this batch (typically equal
        to self.nsamples [original set of values] or 1 [striding]) -
        basically how much to copy one ts in batch dimension.

        Returns:

        batch, Dict[ExampleKeys] -> Tensors[Float]: a batch of
        tensors, each tensor of shape Tensor[NSEQ x nextended x {}]
        (i.e., depends on value of nextended).

        """
        if self.ncoarse is not None:
            return self.__make_one_binned_batch(vals, nextended)
        return self.__make_one_cont_batch(vals, nextended)

    def __make_batch(self, vals_lst, nextended):
        """Given the values and the conditioning windows, turn the data into a
        batch, from which we can call the generator.

        Arguments:

        vals_lst: Tensor[Float], NSERIES x NSEQ: the conditioning
        windows or "priming sequences" that we would like to predict
        forward from.

        nextended: Int, how much to expand the series within this
        batch (typically equal to self.nsamples [original set of
        values] or 1 [striding])

        Returns:

        batch, Dict[ExampleKeys] -> Tensors[Float]: A batch of
        tensors, with each tensor of shape Tensor[NSEQ x
        NSERIES*NSAMPLES x {}].  (trivial series are excluded)

        More details on the batch: tensors in batch come out with
        shape NSEQ x NSERIES*NSAMPLES x {}.  They are stacked so that
        for each element of the sequence (1..NSEQ), we see NSAMPLES
        for the first series, NSAMPLES for the next, etc.  E.g.,
        batch[ExampleKeys.ORIGINALS] might look like:

        tensor([[[ 120.], [ 120.], [ 120.], ..., [ 788.], [ 788.], [
        788.]], ..., [[ 212.], [ 212.], [ 212.], ..., [1080.],
        [1080.], [1080.]]])

        I.e., where 120 is first element of first series, and 788 is
        first element of last series, and 212/1080 is the last element
        of these series.

        Let us call NBATCH = NSERIES*NSAMPLES, so this is the "batch
        dimension" and goes over both series and samples in each
        series.

        """
        all_batches = []
        batch_idx = 0
        for one_vals in vals_lst:
            batch = self.__make_one_ts_batch(one_vals, nextended)
            all_batches.append(batch)
            batch_idx += nextended
        # Note: collating function often used for single examples, but
        # can make batches of batches too:
        batch = CollateUtils.batching_collator(all_batches)
        return batch

    def _make_all_gen(self, vals_lst):
        """Helper to use the generator to make all the samples.

        Arguments:

        vals_lst: Tensor[Float], NSERIES x NSEQ: the conditioning
        windows or "priming sequences" that we would like to predict
        forward from.

        Returns:

        all_gen: Tensor: (NSERIES*NSAMPLES) x GSIZE, the generated data
        for each sample of each element of the batch.

        """
        # Store all the generated ones here:
        all_gens = []
        len_generated = 0
        batch = self.__make_batch(vals_lst, self.nsamples)
        nseries = len(vals_lst)
        vals_lst = vals_lst.repeat(1, self.nsamples).reshape(nseries * self.nsamples, -1)
        while True:
            # Generate nstride ahead
            new_gen = self.gen(batch)
            # Add these nstride ones to our generated set:
            all_gens.append(new_gen)
            # If we now have gsize all_gen, then break:
            len_generated += self.nstride
            if len_generated >= self.gsize:
                break
            raise RuntimeError("Striding not implemented in this version.")
        all_gen = torch.cat(all_gens, dim=1)
        # Trim off if we strode too far:
        all_gen = all_gen[:, :self.gsize]
        return all_gen

    def __make_ptiles(self, all_gen):
        """Helper class to turn all the generated rollouts into
        percentiles.

        Arguments:

        all_gen: Tensor: (NSERIES*NSAMPLES) x GSIZE, the generated data
        for each sample of each element of the batch.

        Returns:

        list[Tensor], where each Tensor is NSERIES x GSIZE.  If
        self.confidence > 0, then these are just [lows, p50s, highs]
        (where each is NSERIES x GSIZE), the percentiles over the
        samples for each series. If self.confidence == -1, then return
        a list of the ptiles at 0.1, 0.2, ..., 0.9 (9 in total), each
        NSERIES x GSIZE in size.

        """
        # Put all_gen into a new format, with a dim for each series:
        all_gen = all_gen.reshape(-1, self.nsamples, self.gsize)
        if self.confidence > 0 and self.confidence < 100:
            ptiles = self.get_ptile_triple(self.confidence, all_gen)
        elif self.confidence == ConfidenceFlags.WQL_QTILES.value:
            ptiles = self.get_wql_qtiles(all_gen)
        elif self.confidence is None:
            ptiles = None
        else:
            msg = f"Undefined or unusable confidence value: {self.confidence}"
            raise RuntimeError(msg)
        # Do this here rather than in else branch to avoid unreachable
        # code OR inconsistent return values (pylint error):
        if ptiles is None:
            raise RuntimeError(f"Invalid confidence value{self.confidence}")
        return ptiles

    @staticmethod
    def get_wql_qtiles(all_gen):
        """Helper to compute the quantiles at 0.1, 0.2, ..., 0.9.

        Arguments:

        all_gen: Tensor: NSERIES x NSAMPLES x GSIZE, the generated
        data for each sample of each element of the batch.

        Returns:

        list[Tensor]: each element NSERIES x GSIZE, nine in total, for
        the quantiles from 0.1, ..., 0.9.

        """
        qtiles = []
        for n in range(1, 10):  # i.e., 1-9 included
            qth = n / 10.0
            qtiles.append(torch.quantile(all_gen, qth, axis=1))
        return qtiles

    @staticmethod
    def get_ptile_triple(confidence, all_gen):
        """Helper to compute the lows/p50s/highs percentiles.

        Arguments:

        confidence: Int, the width of the confidence band, as a
        percentage.

        all_gen: Tensor: NSERIES x NSAMPLES x GSIZE, the generated
        data for each sample of each element of the batch.

        Returns:

        lows, p50s, highs: Tensor: each NSERIES x GSIZE, the
        percentiles over the samples for each series.

        """
        ptiles = []
        ptile_low = (100 - confidence) / 2.0
        ptile_high = 100 - ptile_low
        for qth in [ptile_low, 50, ptile_high]:
            ptiles.append(torch.quantile(all_gen, qth / 100.0, axis=1))
        return ptiles

    def __call__(self, vals_lst):
        """Get the predictions for each set of vals in the given values lists,
        and output some quantiles for each.  Note: we expect the input
        to already be on the device that we are using.  E.g., in
        training/testing, we call `batch_forward` on an evaluator, and
        that will move these values to a device before calling
        make_outputs (which does the generation-based evaluation).

        Arguments:

        vals_lst: Tensor[Float], NSERIES x NSEQ: the conditioning
        windows or "priming sequences" that we would like to predict
        forward from.

        Returns:

        list[Tensor], where each Tensor is NSERIES x GSIZE.  If
        self.confidence > 0, then these are just [lows, p50s, highs]
        (where each is NSERIES x GSIZE), the percentiles over the
        samples for each series. If self.confidence == -1, then return
        a list of the ptiles at 0.1, 0.2, ..., 0.9 (9 in total), each
        NSERIES x GSIZE in size.

        """
        all_gen = self._make_all_gen(vals_lst)
        qtiles = self.__make_ptiles(all_gen)
        return qtiles
