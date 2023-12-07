"""Child of FCDataset specifically for binned input/output
representations.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/bin_dataset.py

import logging
from sutranets.ml.bin_tensor_maker import BinTensorMaker
from sutranets.ml.constants import ExampleKeys
from sutranets.ml.fc_dataset import FCDataset, DatasetArgs
logger = logging.getLogger("sutranets.ml.bin_dataset")


class BinDataset(FCDataset):
    """A dataset that can be used for training forecasting models, using
    binned representations.

    """

    def __init__(self, trace_paths, offsets_fn,
                 coarse_cutoffs, coarse_low, coarse_high, csize, nsize, gsize,
                 trivial_min_nonzero_frac, lagfeat_period, *, dataset_args):
        if dataset_args is None:
            dataset_args = DatasetArgs()
        if lagfeat_period is not None and lagfeat_period >= csize:
            msg = "Can't use lags if not greater than one period in " \
                  f"cwindow, lag_period={lagfeat_period}, csize={csize}"
            raise RuntimeError(msg)
        self.tmaker = BinTensorMaker(coarse_cutoffs,
                                     coarse_low, coarse_high,
                                     nfine_bins=dataset_args.nfine_bins,
                                     nfine2_bins=dataset_args.nfine2_bins,
                                     genmode=dataset_args.genmode,
                                     extremas=dataset_args.extremas,
                                     lagfeat_period=lagfeat_period,
                                     bin_dropout=dataset_args.bin_dropout)
        self.nsize = nsize
        super().__init__(trace_paths, offsets_fn, csize, gsize,
                         trivial_min_nonzero_frac, dataset_args)

    def get_csize_gsize(self):
        """Getter for the csize and gsize

        """
        return self.csize, self.gsize

    def __add_included_fields(self, sample, meta, origs, coarses,
                              fines, fines2):
        """Helper to check whether we're including various things in each
        sample, and, if so, add them.

        """
        if self.dataset_args.include_meta:
            # Prepare for list of lists:
            sample[ExampleKeys.META] = [meta]
        if self.dataset_args.include_originals:
            orig_tensor = self.tmaker.encode_originals(origs)
            sample[ExampleKeys.ORIGINALS] = orig_tensor
        if self.dataset_args.include_coarses:
            coarses_tensor = self.tmaker.encode_originals(coarses)
            sample[ExampleKeys.COARSES] = coarses_tensor
        if self.dataset_args.include_fines:
            fines_tensor = self.tmaker.encode_originals(fines)
            sample[ExampleKeys.FINES] = fines_tensor
        if self.dataset_args.include_fines2:
            fines2_tensor = self.tmaker.encode_originals(fines2)
            sample[ExampleKeys.FINES2] = fines2_tensor

    def _handle_trivial(self, meta, origs):
        """After detecting a TrivialError, we can use this to recover in
        certain cases, otherwise we just return None.

        """
        logger.debug("Skipping trivial series for %s, %s.",
                     meta, origs[:self.csize + 2])
        return None

    def _make_example(self, meta, origs, bin_covars=None):
        """Make an example from the given meta and origs, for a single
        sequence.

        """
        if self.dataset_args.truncate_pt_s is not None:
            origs = self._truncate_origs(meta, origs)
        coarses, fines, fines2, normed_origs, bin_covars = self.tmaker.digitize(
            meta, origs, bin_covars, self.csize, self.nsize, self.trivial_min_nonzero_frac)
        sample = {}
        if not self.dataset_args.targets_only:
            ex_input = self.tmaker.encode_input(
                coarses, fines=fines, fines2=fines2,
                normed=normed_origs, bin_covars=bin_covars)
            sample[ExampleKeys.INPUT] = ex_input
        if not self.dataset_args.genmode:
            ex_target = self.tmaker.encode_target(
                coarses, fines=fines, fines2=fines2, normed=normed_origs)
            self._set_ignored_targets(origs, meta, ex_target, self.csize)
            sample[ExampleKeys.TARGET] = ex_target
        self.__add_included_fields(sample, meta, origs, coarses, fines, fines2)
        return sample
