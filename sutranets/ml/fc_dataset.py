"""Base class for other datasets.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/fc_dataset.py

from abc import abstractmethod
from dataclasses import dataclass
import logging
from torch.utils.data import Dataset
from sutranets.dataprep.data_utils import DataUtils
from sutranets.ml.loss_functions.constants import IGNORE_INDEX
from sutranets.offsets.checkpoint_maker import CheckpointMaker
from sutranets.offsets.window_extractor import WindowExtractor

logger = logging.getLogger("sutranets.ml.fc_dataset")


@dataclass
class DatasetArgs():
    """Encapsulation of optional dataset arguments:

    """
    ncheckpoint: int = -1
    randomize_start: bool = False
    truncate_pt_s: int = None
    loss_end_pt_s: int = None
    loss_start_pt_s: int = None
    high_period_s: int = None
    span_width_s: int = None
    include_meta: bool = False
    include_originals: bool = False
    include_coarses: bool = False
    include_fines: bool = False
    include_fines2: bool = False
    targets_only: bool = False
    cache_set: bool = False
    extremas: bool = False
    genmode: bool = False
    nfine_bins: bool = None
    nfine2_bins: bool = None
    bin_dropout: float = None


class FCDataset(Dataset):
    """Dataset used for training forecasting models.

    """

    def __init__(self, trace_paths, offsets_fn, csize, gsize,
                 trivial_min_nonzero_frac, dataset_args):
        if dataset_args is None:
            self.dataset_args = DatasetArgs()
        else:
            self.dataset_args = dataset_args
        self.csize = csize
        self.gsize = gsize
        self.trivial_min_nonzero_frac = trivial_min_nonzero_frac
        if self.dataset_args.cache_set:
            self.example_cache = {}
        else:
            self.example_cache = None
        self.__validate()
        if trace_paths is None:
            self.window_extractor = None
            self.cp_maker = None
            self.offset = None
        else:
            self.window_extractor = WindowExtractor(trace_paths)
            self.cp_maker = CheckpointMaker(
                offsets_fn, ncheckpoint=self.dataset_args.ncheckpoint,
                randomize_start=self.dataset_args.randomize_start)
            self.offsets = next(self.cp_maker.yield_checkpoints())

    def advance_next_epoch(self):
        """Resets to a (potentially) different set of offsets.

        """
        self.offsets = next(self.cp_maker.yield_checkpoints())

    def __validate(self):
        """Check for certain invalid combos of options."""
        loss_start = self.dataset_args.loss_start_pt_s
        loss_end = self.dataset_args.loss_end_pt_s
        if loss_start is not None and loss_end is not None:
            if loss_end <= loss_start:
                raise RuntimeError("loss_end must be > loss_start")
        if self.dataset_args.truncate_pt_s is not None:
            if loss_start is not None:
                raise RuntimeError("Don't support both truncating AND loss start pt.")
            if loss_end is not None:
                raise RuntimeError("Don't support both truncating AND loss end pt.")
        self.__target_only_checks(loss_start, loss_end)

    def __target_only_checks(self, loss_start, loss_end):
        """Just a helper to make validate() itself less complex."""
        if self.dataset_args.targets_only or loss_start is not None \
           or loss_end is not None:
            if self.dataset_args.genmode:
                raise RuntimeError("Can't use genmode + target-based options (no targs)")
        if self.dataset_args.targets_only:
            if self.dataset_args.include_coarses:
                raise RuntimeError("Coarses can't be kept if doing targets_only")
            if self.dataset_args.include_fines:
                raise RuntimeError("Fines can't be kept if doing targets_only")
            if self.dataset_args.include_fines2:
                raise RuntimeError("Fines can't be kept if doing targets_only")

    def _truncate_origs(self, meta, origs):
        """Used when dividing our data into train/dev/test set: truncate data
        in our orig arrays at our truncate_pt_s point (truncate this
        point, and all subsequent points, inclusive).

        """
        ntruncated = DataUtils.timestamp_to_nexclude_pts(
            meta, self.dataset_args.truncate_pt_s,
            high_period_s=self.dataset_args.high_period_s)
        if ntruncated > 0:
            if origs is not None:
                origs = origs[:-1 * ntruncated]
        if not origs.nelement():
            msg = "Pointless execution 3: " \
                  "origs has no elements."
            logger.error(msg)
            logger.error(meta)
            raise RuntimeError(msg)
        return origs

    @staticmethod
    def __highp_err(meta, highp, msg):
        """Raise an error with the given msg only if we are at highp (or highp
        is None so we are definitely at highp because there is only
        one p).

        """
        if highp is None or DataUtils.get_sampling_period_s(meta) == highp:
            full_msg = f"Pointless execution: {msg} at {meta}"
            raise RuntimeError(full_msg)

    def _set_ignored_targets(self, origs, meta, targets, csize):
        """Modifies targets to be ignored from the loss (in training or
        testing).

        """
        highp = self.dataset_args.high_period_s
        spanw = self.dataset_args.span_width_s
        targets[origs[2:].isnan()] = IGNORE_INDEX
        loss_start = self.dataset_args.loss_start_pt_s
        loss_end = self.dataset_args.loss_end_pt_s
        if loss_start is not None:
            ninclude_loss = DataUtils.timestamp_to_ninclude_pts(
                meta, loss_start, high_period_s=highp, span_width_s=spanw)
            if ninclude_loss == 0:
                self.__highp_err(meta, highp, "Loss starts after entire sequence")
        else:
            ninclude_loss = len(targets)
        gen_size = max(len(targets) - csize, 0)
        if gen_size < ninclude_loss:
            ninclude_loss = gen_size
        if ninclude_loss > 0:
            targets[:-1 * ninclude_loss] = IGNORE_INDEX
        else:
            targets[:] = IGNORE_INDEX
        if loss_end is not None:
            nexclude_loss = DataUtils.timestamp_to_nexclude_pts(
                meta, loss_end, high_period_s=highp)
        else:
            nexclude_loss = 0
        if nexclude_loss > 0:
            targets[-1 * nexclude_loss:] = IGNORE_INDEX
        if not self.dataset_args.genmode and \
           (targets == IGNORE_INDEX).all():
            self.__highp_err(meta, highp, "All targets are ignored")

    def __len__(self):
        """Each offset is its own example, so this is just the number we have.

        """
        return len(self.offsets)

    @abstractmethod
    def _make_example(self, meta, origs):
        """Make an example from the given meta and origs, for a single
        sequence.

        """

    def __getitem__(self, idx):
        """Return an example using the offset at idx and the input files."""
        if idx < 0 or idx >= self.__len__():
            raise IndexError
        if self.example_cache is not None and idx in self.example_cache:
            return self.example_cache[idx]
        offset = self.offsets[idx]
        meta, origs = self.window_extractor(offset, self.csize, self.gsize)
        example = self._make_example(meta, origs)
        if self.example_cache is not None and idx not in self.example_cache:
            self.example_cache[idx] = example
        return example
