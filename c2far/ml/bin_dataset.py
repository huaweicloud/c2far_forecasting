"""Child of FCDataset specifically for binned input/output
representations.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/bin_dataset.py

from c2far.ml.constants import ExampleKeys
from c2far.ml.fc_dataset import FCDataset, DatasetArgs
from c2far.ml.bin_tensor_maker import BinTensorMaker


class BinDataset(FCDataset):
    """A dataset that can be used for training forecasting models, using
    binned representations.

    """

    def __init__(self, vcpus_fn, mem_fn, offsets_fn, coarse_cutoffs,
                 coarse_low, coarse_high, csize, nsize, gsize,
                 dataset_args):
        """Initialize the dataset class.

        Args:

        vcpus_fn: String, file containing original vcpus ts data.

        mem_fn: String, file containing original memory ts data.

        offsets_fn: String, the file containing all the offsets into
        the time series data.

        coarse_cutoffs: Tensor[NBINS], list of coarse cutoffs.

        coarse_low: Float, the lowest implicit boundary for the coarse
        cutoffs.

        coarse_high: Float, the highest implicit boundary for the
        coarse cutoffs.

        csize: Int, the number of samples from the beginning of the
        sequence to deem part of the 'conditioning window' and to NOT
        compute loss over.  n.b. - this is done AFTER we consume
        samples to create delta and for last target.

        gsize: Int, number of samples to be in 'generation window'.

        dataset_args: DatasetArgs, see fc_dataset.py. `None` for all defaults.

        """
        # We do this check now (and superfluously in super) because we
        # may need default args to make the tmaker.
        if dataset_args is None:
            dataset_args = DatasetArgs()
        tmaker = BinTensorMaker(coarse_cutoffs, coarse_low,
                                coarse_high,
                                nfine_bins=dataset_args.nfine_bins,
                                nfine2_bins=dataset_args.nfine2_bins,
                                genmode=dataset_args.genmode,
                                extremas=dataset_args.extremas)
        super().__init__(vcpus_fn, mem_fn, offsets_fn, csize, nsize,
                         gsize, tmaker, dataset_args)

    def _make_example(self, meta, origs, csize):
        """Make an example from the given meta and origs, for a single
        sequence.

        """
        coarses, fines, fines2, normed_origs = self.tmaker.digitize(
            origs, self.csize, self.nsize)
        ex_input = self.tmaker.encode_input(
            coarses, fines=fines, fines2=fines2, normed=normed_origs)
        # Encode targets is also done by the tmaker, except in
        # generation mode, where there are no targets:
        if not self.dataset_args.genmode:
            ex_target = self.tmaker.encode_target(
                coarses, fines=fines, fines2=fines2, normed=normed_origs)
            self._set_ignored_targets(origs, meta, ex_target, csize)
        else:
            ex_target = None
        sample = {ExampleKeys.INPUT: ex_input,
                  ExampleKeys.TARGET: ex_target}
        self.__add_included_fields(sample, origs, coarses)
        return sample

    def __add_included_fields(self, sample, origs, coarses):
        """Helper to check whether we're including various things in each
        sample, and, if so, add them.

        Returns:

        Nothing, modifies 'sample' in-place.

        """
        # For originals and bin idxs, store as NSEQ+2x1x1 tensors:
        if self.dataset_args.include_originals:
            orig_tensor = self.tmaker.encode_originals(origs)
            sample[ExampleKeys.ORIGINALS] = orig_tensor
        if self.dataset_args.include_coarses:
            coarses_tensor = self.tmaker.encode_originals(coarses)
            sample[ExampleKeys.COARSES] = coarses_tensor
