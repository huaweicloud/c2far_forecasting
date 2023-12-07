"""Given a single offset, this class provides a method to extract a
window in the time series (meta, origs), that can then be used for
example creation and training.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/offsets/window_extractor.py
import os
import torch
from sutranets.dataprep.data_utils import DataUtils
from sutranets.dataprep.io_utils import get_meta_vals
DUMMY_VAL = float('nan')


class TraceIndexError(Exception):
    """Base class for exceptions when we try to index a trace file but it
    goes beyond the number of traces in our lists (we can't raise
    regular IndexError here because the Dataset class catches it).

    """


class WindowExtractor():
    """Makes the example for a single offset, based on the original offset
    file schema and linecache approach.

    """
    def __init__(self, trace_paths):
        self.trace_paths = trace_paths
        for fname in self.trace_paths:
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
            origs += [DUMMY_VAL] * (win_size - len(origs))
        if len(origs) != win_size:
            msg = f"Not enough origs ({len(origs)}) " \
                  f"for win_size ({win_size}): {origs}"
            raise RuntimeError(msg)
        origs = torch.tensor(origs)
        return origs

    def __call__(self, offset, csize, gsize):
        """Take in a structured offset and unpack it, use it to retrieve a
        window from the data.

        """
        last_ds, line_num, target_idx, trace_idx = offset
        try:
            target_fn = self.trace_paths[trace_idx]
        except IndexError as exp:
            raise TraceIndexError from exp
        meta, origs = get_meta_vals(target_fn, line_num)
        origs = self.__extract_origs(target_idx, origs, csize, gsize)
        DataUtils.set_last_s(meta, last_ds)
        return meta, origs
