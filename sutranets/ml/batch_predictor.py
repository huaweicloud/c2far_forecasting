"""Standalone, self-contained predictor that can take in a batch of
conditioning sequences and generate all the samples for each of them
for the future.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/batch_predictor.py
import logging
import torch
from sutranets.dataprep.data_utils import DataUtils
from sutranets.ml.bin_tensor_maker import BinTensorMaker
from sutranets.ml.constants import ExampleKeys
from sutranets.ml.fc_digitizer import FCDigitizer
from sutranets.ml.generator import Generator
from sutranets.ml.joint_generator import JointGenerator
from sutranets.ml.triple_generator import TripleGenerator

logger = logging.getLogger("sutranets.ml.batch_predictor")


class BatchPredictor():
    """Init with an LSTM evaluator, different coarse/delta maps, and other
    key parameters.  Then, give a list of conditioning sequences, get
    nsamples of the future for each of them.

    """

    def __init__(self, lstm_evaluator, coarse_cutoffs, coarse_low, coarse_high,
                 *, csize, nsize, nstride, gsize, trivial_min_nonzero_frac, nsamples, device,
                 confidence, bdecoder, nfine_bins, nfine2_bins, extremas, lagfeat_period):
        self.nfine = nfine_bins
        self.nfine2 = nfine2_bins
        self.nsize = nsize
        self.nstride = nstride
        self.csize = csize
        self.gsize = gsize
        self.trivial_min_nonzero_frac = trivial_min_nonzero_frac
        self.device = device
        self.nsamples = nsamples
        self.confidence = confidence
        self.bdecoder = bdecoder
        self.ncoarse = len(coarse_cutoffs) + 1
        self.tmaker, self.digitizer, self.gen = self.__make_binned_objects(
            lstm_evaluator, coarse_cutoffs, coarse_low, coarse_high,
            extremas, lagfeat_period)

    def __make_binned_objects(self, lstm_evaluator, coarse_cutoffs,
                              coarse_low, coarse_high, extremas,
                              lagfeat_period):
        """Helper to handle making the binned tmaker/digitizer/generators."""
        coarse_cutoffs = coarse_cutoffs.to(self.device)
        tmaker = BinTensorMaker(coarse_cutoffs,
                                coarse_low, coarse_high, nfine_bins=self.nfine,
                                nfine2_bins=self.nfine2, genmode=True, extremas=extremas,
                                lagfeat_period=lagfeat_period)
        digitizer = FCDigitizer(coarse_cutoffs, coarse_low, coarse_high,
                                nfine_bins=self.nfine, nfine2_bins=self.nfine2,
                                extremas=extremas, check_trivial=False)
        if self.nfine is None:
            gen = Generator(self.device, lstm_evaluator,
                            coarse_cutoffs, coarse_low, coarse_high, self.nsize, ngen=self.nstride,
                            bdecoder=self.bdecoder, extremas=extremas, lagfeat_period=lagfeat_period)
        elif self.nfine2 is None:
            gen = JointGenerator(self.ncoarse, self.nfine, self.device, lstm_evaluator,
                                 coarse_cutoffs, coarse_low, coarse_high, self.nsize,
                                 ngen=self.nstride, bdecoder=self.bdecoder, extremas=extremas,
                                 lagfeat_period=lagfeat_period)
        else:
            gen = TripleGenerator(self.ncoarse, self.nfine, self.nfine2, self.device, lstm_evaluator,
                                  coarse_cutoffs, coarse_low, coarse_high, self.nsize,
                                  ngen=self.nstride, bdecoder=self.bdecoder, extremas=extremas,
                                  lagfeat_period=lagfeat_period)
        return tmaker, digitizer, gen

    def __str__(self):
        """Provide some quick info about what kind of predictor this is."""
        if self.nfine is None:
            mystr = f"CoarsePred.c={self.ncoarse}"
        elif self.nfine2 is None:
            mystr = f"JointPred.c={self.ncoarse}.f={self.nfine}"
        else:
            mystr = f"TriplePred.c={self.ncoarse}.f={self.nfine}.f2={self.nfine2}"
        mystr += f".{self.bdecoder.value}"
        mystr += f".n{self.nsamples}.conf{self.confidence}.csize={self.csize}.gsize={self.gsize}"
        return mystr

    def __digitize(self, origs, cwindow_bin_covars):
        """Digitize values into our usual encoding.

        """
        meta = f"{self.__class__.__name__}-series"
        assert len(origs) == self.csize + 2, \
            f"Unexpected arr length: {len(origs)} vs. {self.csize + 2}"
        coarses, fines, fines2, normed, cwindow_bin_covars = self.digitizer(
            meta, origs, cwindow_bin_covars, self.csize, self.nsize, self.trivial_min_nonzero_frac)
        return coarses, fines, fines2, normed, cwindow_bin_covars

    def __make_binned_batch(self, vals, nextended, cwindow_bin_covars):
        """Given the values for a set of time series, turn the data into a
        batch, for BINNED prediction.

        """
        origs = vals
        coarses, fines, fines2, normed, cwindow_bin_covars = self.__digitize(origs, cwindow_bin_covars)
        inputs = self.tmaker.encode_input(coarses, fines=fines, fines2=fines2,
                                          normed=normed, bin_covars=cwindow_bin_covars)
        if nextended > 1:
            inputs = inputs.repeat_interleave(nextended, dim=1)
            origs = origs.repeat_interleave(nextended, dim=1)
            coarses = coarses.repeat_interleave(nextended, dim=1)
            if fines is not None:
                fines = fines.repeat_interleave(nextended, dim=1)
            if fines2 is not None:
                fines2 = fines2.repeat_interleave(nextended, dim=1)
        origs = origs.unsqueeze(2)
        coarses = coarses.unsqueeze(2)
        if fines is not None:
            fines = fines.unsqueeze(2)
        if fines2 is not None:
            fines2 = fines2.unsqueeze(2)
        batch = {ExampleKeys.INPUT: inputs, ExampleKeys.ORIGINALS: origs, ExampleKeys.COARSES: coarses,
                 ExampleKeys.FINES: fines, ExampleKeys.FINES2: fines2}
        return batch

    def __get_trivs(self, seq_order, cwindow_bin_covars):
        """Filter out trivial sequences, but return their indices.

        """
        triv_mask = DataUtils.torch_batch_is_trivial(seq_order, self.trivial_min_nonzero_frac)
        non_trivs = seq_order[:, torch.logical_not(triv_mask)]
        triv_idxs = torch.where(triv_mask)[0].reshape(-1).tolist()
        if non_trivs.size(1) == 0:
            non_trivs = None
        if cwindow_bin_covars is not None:
            cwindow_bin_covars.filter_trivials(triv_mask)
        return non_trivs, cwindow_bin_covars, triv_idxs, triv_mask

    def __make_batch(self, vals_lst, nextended, cwindow_bin_covars):
        """Given the values and the conditioning windows, turn the data into a
        batch, from which we can call the generator.

        """
        seq_order = vals_lst.transpose(0, 1)
        non_trivs, nontriv_bin_covars, trivial_batch_idxs, triv_mask = self.__get_trivs(
            seq_order, cwindow_bin_covars)
        if non_trivs is None:
            return None, trivial_batch_idxs, triv_mask
        batch = self.__make_binned_batch(non_trivs, nextended, nontriv_bin_covars)
        return batch, trivial_batch_idxs, triv_mask

    def __remake_batch(self, vals_lst, new_gen, cwindow_bin_covars):
        """In preparation for another round of generation, create a new batch
        by extending the vals_lst with values generated during the
        previous generation.

        """
        vals_lst = torch.cat([vals_lst, new_gen], dim=1)
        keep_amt = self.csize + 2
        vals_lst = vals_lst[:, -keep_amt:]
        batch, trivial_batch_idxs, triv_mask = self.__make_batch(vals_lst, 1, cwindow_bin_covars)
        return batch, trivial_batch_idxs, vals_lst, triv_mask

    @staticmethod
    def __repair_trivials(nstride, new_gen, trivial_batch_idxs, vals_lst):
        """Return excluded trivial elements from the batch back in.

        """
        if not trivial_batch_idxs:
            return new_gen
        ntrivials = len(trivial_batch_idxs)
        logger.debug("%d trivial seqs", ntrivials)
        if new_gen is None:
            new_gen = torch.tensor([], device=vals_lst.device)
        for i in trivial_batch_idxs:
            last_orig = vals_lst[i][-1]
            new_pred = last_orig.repeat(1, nstride)
            new_gen = torch.cat([new_gen[:i], new_pred, new_gen[i:]], 0)
        return new_gen

    def __digitize_gen_bin_covars(self, batch, triv_mask, gen_bin_covars):
        """Helper to take care of digitizing the gen_bin_covars wrt. the current origs"""
        if gen_bin_covars is None:
            return None
        if batch is None:
            return None
        gen_bin_covars.filter_trivials(triv_mask)
        origs = batch[ExampleKeys.ORIGINALS]
        origs = origs.squeeze(2)
        meta = None
        _, _, _, _, gen_bin_covars = self.digitizer(
            meta, origs, gen_bin_covars, self.csize, self.nsize, self.trivial_min_nonzero_frac)
        return gen_bin_covars

    def __extend_initials(self, vals_lst, trivial_batch_idxs, triv_mask):
        """Repeat the values in vals_lst, trivial_batch_idxs, and triv_mask to
        reflect all samples.

        """
        vals_lst = vals_lst.repeat_interleave(self.nsamples, dim=0)
        new_idxs = []
        for idx in trivial_batch_idxs:
            new_idxs += range(idx*self.nsamples, idx*self.nsamples + self.nsamples)
        trivial_batch_idxs = new_idxs
        triv_mask = triv_mask.repeat_interleave(self.nsamples)
        return vals_lst, trivial_batch_idxs, triv_mask

    def __call__(self, vals_lst, cwindow_bin_covars=None,
                 gen_bin_covars=None):
        """Yield the predictions for each set of vals in the given values
        lists.

        """
        len_generated = 0
        batch, trivial_batch_idxs, triv_mask = self.__make_batch(vals_lst, self.nsamples, cwindow_bin_covars)
        vals_lst, trivial_batch_idxs, triv_mask = self.__extend_initials(vals_lst, trivial_batch_idxs, triv_mask)
        while True:
            gen_bin_covars = self.__digitize_gen_bin_covars(batch, triv_mask, gen_bin_covars)
            curr_stride = []
            for generated in self.gen(batch, gen_bin_covars):
                if trivial_batch_idxs:
                    generated = self.__repair_trivials(1, generated, trivial_batch_idxs, vals_lst)
                yield generated
                gen_bin_covars = self.__digitize_gen_bin_covars(batch, triv_mask, gen_bin_covars)
                len_generated += 1
                if len_generated >= self.gsize:
                    return
                curr_stride.append(generated)
            stride_gen = torch.cat(curr_stride, dim=1)
            batch, trivial_batch_idxs, vals_lst, triv_mask = self.__remake_batch(
                vals_lst, stride_gen, cwindow_bin_covars)
