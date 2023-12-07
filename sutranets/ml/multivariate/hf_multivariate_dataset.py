"""Child of FCDataset specifically for sub-series-based multivariate
dataset.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/multivariate/hf_multivariate_dataset.py
from sutranets.dataprep.aggregate_seqs import collect_and_agg, AggFunction
from sutranets.ml.bin_covariates import BinCovariates
from sutranets.ml.bin_dataset import BinDataset
from sutranets.ml.constants import ExampleKeys
from sutranets.ml.fc_dataset import FCDataset
from sutranets.ml.fc_digitizer import TrivialError
from sutranets.ml.multivariate.multi_freq_mapper import MultiFreqMapper
from sutranets.ml.multivariate.tensor_list import TensorList


class HFMultivariateDataset(FCDataset):
    """A dataset that internally divides a series into sub-series and
    creates composite examples comprised of tensors for all of them.

    """

    def __init__(self, trace_paths, offsets_fn,
                 coarse_cutoffs, coarse_low, coarse_high, csize,
                 gsize, trivial_min_nonzero_frac, lagfeat_period, *, dataset_args,
                 sub_csize, nsub_series, mv_no_prevs, mv_backfill):
        self.dargs = dataset_args
        if self.dargs is None or self.dargs.high_period_s is None:
            msg = "Dataset args, with a defined high_period_s, required for multivariate."
            raise RuntimeError(msg)
        self.dargs.span_width_s = self.dargs.high_period_s
        super().__init__(trace_paths, offsets_fn, csize, gsize, trivial_min_nonzero_frac, self.dargs)
        self.__validate(gsize, lagfeat_period, nsub_series)
        self.nsub_series = nsub_series
        sub_nsize = self.sub_csize = sub_csize
        self.sub_gsize = MultiFreqMapper.divide_and_check(gsize, nsub_series)
        self.include_prevs = not mv_no_prevs
        self.backfill = mv_backfill
        sub_lagfp = MultiFreqMapper.divide_and_check(lagfeat_period, nsub_series)
        self.bin_dataset = BinDataset(None, None, coarse_cutoffs, coarse_low,
                                      coarse_high, self.sub_csize, sub_nsize, self.sub_gsize,
                                      trivial_min_nonzero_frac, sub_lagfp, dataset_args=self.dargs)

    def __validate(self, gsize, lagfeat_period, nsub_series):
        msg_prefix = f"{nsub_series} does not divide evenly into"
        if gsize % nsub_series:
            raise RuntimeError(f"{msg_prefix} gsize {gsize}")
        if lagfeat_period is not None and lagfeat_period % nsub_series:
            raise RuntimeError(f"{msg_prefix} lagfeat_period {lagfeat_period}")

    def __compose_samples(self, all_samples):
        """Compose the different samples into a single unit, compressing all
        values for each key into a single TensorList.

        """
        composite = {}
        for samp in all_samples:
            for key, value in samp.items():
                if key not in composite:
                    if key == ExampleKeys.META:
                        composite[key] = [[]]
                    else:
                        composite[key] = TensorList()
                if key == ExampleKeys.META:
                    composite[key][0].append(value[0])
                else:
                    composite[key].append(value)
        return composite

    def _make_example(self, meta, origs):
        """Make an example from the given meta and origs, creating a
        multivariate series from a single HF sequence.

        """
        all_sub_metas, all_sub_origs = [], []
        if self.backfill:
            ordered_offsets = reversed(range(self.nsub_series))
        else:
            ordered_offsets = range(self.nsub_series)
        for sub_offset in ordered_offsets:
            sub_meta, sub_orig = collect_and_agg(meta, origs, self.nsub_series, self.sub_csize,
                                                 self.sub_gsize, AggFunction.SUBSERIES, sub_offset)
            all_sub_metas.append(sub_meta)
            all_sub_origs.append(sub_orig)
        bin_covars = BinCovariates(raw_lst=all_sub_origs, include_prevs=self.include_prevs)
        all_samples = []
        for series_idx, (sub_meta, sub_orig) in enumerate(zip(all_sub_metas, all_sub_origs)):
            bin_covars.set_target_level(series_idx)
            try:
                sample = self.bin_dataset._make_example(sub_meta, sub_orig, bin_covars)
            except TrivialError:
                sample = self.bin_dataset._handle_trivial(sub_meta, sub_orig)
                if sample is None:
                    return None
            all_samples.append(sample)
        sample = self.__compose_samples(all_samples)
        return sample
