"""Child of FCDataset specifically for continuous input/output
representations.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/continuous/cont_dataset.py

from c2far.dataprep.data_utils import DataUtils
from c2far.ml.constants import ExampleKeys
from c2far.ml.fc_dataset import FCDataset
from c2far.ml.continuous.cont_tensor_maker import ContTensorMaker


class ContDataset(FCDataset):
    """A dataset that can be used for training forecasting models, using
    continuous representations.

    """

    def __init__(self, vcpus_fn, mem_fn, offsets_fn, csize, nsize,
                 gsize, dataset_args):
        """See the parent for full description of arguments."""
        tmaker = ContTensorMaker(genmode=dataset_args.genmode)
        super().__init__(vcpus_fn, mem_fn, offsets_fn, csize, nsize,
                         gsize, tmaker, dataset_args)

    def _make_example(self, meta, origs, csize):
        """Make an example from the given meta and origs, for a single
        sequence.

        """
        # For continuous ones, we need to norm the originals:
        normed = DataUtils.torch_norm(self.csize, self.nsize, origs)
        ex_input = self.tmaker.encode_input(normed)
        # Encode targets is also done by the tmaker, except in
        # generation mode, where there are no targets:
        if not self.dataset_args.genmode:
            ex_target = self.tmaker.encode_target(normed)
            self._set_ignored_targets(origs, meta, ex_target, csize)
        else:
            ex_target = None
        sample = {ExampleKeys.INPUT: ex_input,
                  ExampleKeys.TARGET: ex_target}
        # For originals and bin idxs, store as NSEQ+2x1x1 tensors:
        if self.dataset_args.include_originals:
            orig_tensor = self.tmaker.encode_originals(origs)
            sample[ExampleKeys.ORIGINALS] = orig_tensor
        return sample
