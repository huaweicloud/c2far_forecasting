"""Given a single offset, this class provides a method to extract a
window in the time series (meta, origs), that can then be used for
example creation and training.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/offsets/window_extractor.py
import os
import torch
from c2far.dataprep.data_utils import DataUtils
from c2far.dataprep.io_utils import get_meta_vals
DUMMY_VAL = float('nan')


class WindowExtractor():
    """Makes the example for a single offset, based on the original offset
    file schema and linecache approach.

    """
    def __init__(self, vcpus_fn, mem_fn):
        """Initialize the window_extractor to know about the vcpus and memory
        files.

        Arguments:

        vcpus_fn: String, file containing original vcpus ts data.

        mem_fn: String, file containing original memory ts data.

        """
        self.vcpus_fn = vcpus_fn
        self.mem_fn = mem_fn
        for fname in [self.vcpus_fn, self.mem_fn]:
            if not os.path.exists(fname):
                raise FileNotFoundError(f"File {fname} does not exist")

    @staticmethod
    def __extract_origs(target_idx, origs, csize, gsize):
        """Get a window at the given offset, as a tensor. Handle case where
        window spans beyond actual data.

        """
        win_size = csize + gsize + 2
        origs = origs[target_idx:target_idx + win_size]
        if len(origs) < win_size:
            # Pad the originals so we know these are missing:
            origs += [DUMMY_VAL] * (win_size - len(origs))
        # pylint: disable=not-callable
        origs = torch.tensor(origs)
        return origs

    def __call__(self, offset, csize, gsize):
        """Take in a structured offset and unpack it, use it to retrieve a
        window from the data.

        """
        last_ds, line_num, target_idx, res = offset
        if res == 0:
            target_fn = self.vcpus_fn
        else:
            target_fn = self.mem_fn
        meta, origs = get_meta_vals(target_fn, line_num)
        origs = self.__extract_origs(target_idx, origs, csize, gsize)
        DataUtils.set_last_s(meta, last_ds)
        return meta, origs
