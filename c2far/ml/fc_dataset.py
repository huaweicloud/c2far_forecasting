"""Subclass Torch's ABC Dataset and provide a base class for other
datasets. Each with methods to __getitem__() and __len__() so we can
retrieve examples smoothly during training - for our forecasting data.

Rather than training on all the windows at once, this dataset can
divide the offsets into epochs and then len() and [] will work just
within the offsets for that epoch.  Then, you can call
'advance_next_epoch()' to move to the next one.  We call these
"checkpoints".

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/fc_dataset.py

from abc import abstractmethod
from dataclasses import dataclass
import logging
from torch.utils.data import Dataset
from c2far.dataprep.data_utils import DataUtils
from c2far.ml.loss_functions.constants import IGNORE_INDEX
from c2far.offsets.checkpoint_maker import CheckpointMaker
from c2far.offsets.window_extractor import WindowExtractor

logger = logging.getLogger("c2far.ml.fc_dataset")


@dataclass
class DatasetArgs():
    """Encapsulation of optional dataset arguments:

    ncheckpoint: Int, how many examples to process before exhausting
    the getitem iterator (allows us to check progress).  Pass '-1' to
    default to all the offsets, pass 'None' to infer.

    randomize_start: Boolean, whether to randomize our starting
    position in the offsets. No reason not to do this each time, but I
    keep it False by default so that UTs work without needing changes.

    loss_start_pt_s: Int, the first value that we SHOULD include in
    the loss - include all after (and including) this point.

    loss_end_pt_s: Int, the first value that we should NOT include in
    the loss - exclude all after (and including) this point.

    include_originals: Boolean, whether to include the originals for
    each line with each example.

    include_coarses: Boolean, whether to include encoded `coarses` for
    each line with each example - is useful for some loss functions.

    cache_set: Boolean, if True we cache examples after creating them.

    extremas: bool = False, whether to use parameterized functions in
    the extreme bins.

    genmode: Boolean, if True, indicates you're not training/
    evaluating, meaning: (1) No targets, (2) it's okay to not have any
    generation WINDOW, and (3) we encode ALL of the input sequence as
    input (we don't save the last value as the last target).

    nfine_bins: bool = None, how many fine bins to use within each
    coarse, if doing joint generation.

    nfine2_bins: bool = None, how many fine2 bins to use within each
    fine, if doing triple-generation.

    """
    # By default, get ALL the offsets:
    ncheckpoint: int = -1
    randomize_start: bool = False
    loss_end_pt_s: int = None
    loss_start_pt_s: int = None
    include_originals: bool = False
    include_coarses: bool = False
    cache_set: bool = False
    extremas: bool = False
    genmode: bool = False
    nfine_bins: bool = None
    nfine2_bins: bool = None


class FCDataset(Dataset):
    """Abstract base class for datasets that can be used for training
    forecasting models.

    """

    def __init__(self, vcpus_fn, mem_fn, offsets_fn, csize, nsize,
                 gsize, tmaker, dataset_args):
        """Initialize the dataset class.

        Args:

        vcpus_fn: String, file containing original vcpus ts data.

        mem_fn: String, file containing original memory ts data.

        offsets_fn: String, the file containing all the offsets into
        the time series data.

        csize: Int, the number of samples from the beginning of the
        sequence to deem part of the 'conditioning window' and to NOT
        compute loss over.  n.b. - this is done AFTER we consume
        samples to create delta and for last target.

        gsize: Int, number of samples to be in 'generation window'.

        tmaker: TensorMaker (e.g. BinTensorMaker, or ContTensorMaker)
        as used in making examples, returning info, etc.

        dataset_args: DatasetArgs, see above. `None` for all defaults.

        """
        self.tmaker = tmaker
        if dataset_args is None:
            self.dataset_args = DatasetArgs()
        else:
            self.dataset_args = dataset_args
        self.window_extractor = WindowExtractor(vcpus_fn, mem_fn)
        self.csize = csize
        self.nsize = nsize
        self.gsize = gsize
        if self.dataset_args.cache_set:
            self.example_cache = {}
        else:
            self.example_cache = None
        self.__validate()
        self.cp_maker = CheckpointMaker(
            offsets_fn, ncheckpoint=self.dataset_args.ncheckpoint,
            randomize_start=self.dataset_args.randomize_start)
        self.offsets = next(self.cp_maker.yield_checkpoints())

    def get_ninput(self):
        """In this dataset, how many dimensions are in the input?

        """
        return self.tmaker.get_ninput()

    def get_noutput(self):
        """In this dataset, how many dimensions are in the output?

        """
        return self.tmaker.get_noutput()

    def advance_next_epoch(self):
        """When you call this, it resets to a (potentially) different set of
        offsets.

        """
        self.offsets = next(self.cp_maker.yield_checkpoints())

    def __validate(self):
        """Check for certain invalid combos of options."""
        # If using both, these must be ordered properly:
        loss_start = self.dataset_args.loss_start_pt_s
        loss_end = self.dataset_args.loss_end_pt_s
        if loss_start is not None and loss_end is not None:
            if loss_end <= loss_start:
                raise RuntimeError("loss_end must be > loss_start")
        self.__target_only_checks(loss_start, loss_end)

    def __target_only_checks(self, loss_start, loss_end):
        """Just a helper to make validate() itself less complex."""
        if loss_start is not None \
           or loss_end is not None:
            if self.dataset_args.genmode:
                raise RuntimeError("Can't use genmode + target-based options (no targs)")

    def _set_ignored_targets(self, origs, meta, targets, csize):
        """Some targets will be ignored from the loss (in training or testing)
        for various reasons: (1) there is a loss_end_pt: we may have
        to ignore the targets at the end of the sequence, (2) there is
        a loss_start_pt, so we ignore the targets before the given
        loss start point, (3) there is a csize, so we ignore the
        targets in the conditioning window (very common), or (4) our
        offset took part of the generation window past the end of the
        values, so we have to ignore the DUMMY_VAL originals.

        Modifies targets in place.

        """
        # First, ignore dummy originals - remember, these are nans
        # (see DUMMY_VAL in window_extractor.py):
        targets[origs[2:].isnan()] = IGNORE_INDEX
        loss_start = self.dataset_args.loss_start_pt_s
        loss_end = self.dataset_args.loss_end_pt_s
        if loss_start is not None:
            ninclude_loss = DataUtils.timestamp_to_num_pts(
                meta, loss_start)
            if ninclude_loss is None:
                raise RuntimeError("Pointless execution 1: "
                                   "Loss starts after entire sequence.")
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
            nexclude_loss = DataUtils.timestamp_to_num_pts(meta, loss_end)
        else:
            nexclude_loss = None
        if nexclude_loss is not None:
            targets[-1 * nexclude_loss:] = IGNORE_INDEX
        if not self.dataset_args.genmode and \
           (targets == IGNORE_INDEX).all():
            # Only an error when not generating. Need not be fatal,
            # but indicates other probs in our pipeline, so raise:
            msg = "Pointless execution 2: All targets are ignored. "
            logger.error("%s: %s %d %s", msg, str(loss_start), csize, meta)
            logger.error("targets: %s, %s", targets.shape, targets)
            raise RuntimeError(msg)

    def __len__(self):
        """Each offset is its own example, so this is just the number we have.

        """
        return len(self.offsets)

    @abstractmethod
    def _make_example(self, meta, origs, csize):
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
        example = self._make_example(meta, origs, self.csize)
        if self.example_cache is not None and idx not in self.example_cache:
            self.example_cache[idx] = example
        return example
