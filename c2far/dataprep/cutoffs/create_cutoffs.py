"""Helper to dynamically create the coarse cutoffs.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# create_cutoffs.py

import logging
import torch
from c2far.dataprep.cutoffs.linear_cutoffs import get_coarse_cutoffs
from c2far.ml.constants import BinStrategy
logger = logging.getLogger("c2far.dataprep.cutoffs.create_cutoffs")


class CreateCutoffs():
    """Utilities for creating cutoffs."""

    @staticmethod
    def get_coarse_cutoffs(args):
        """Create coarse cutoffs based on the given arguments.

        Arguments:

        args: NamedTuple CLI args, including, e.g., ncoarse_bins.

        Returns:

        coarse_cutoffs: Tensor[NBINS], list of coarse cutoffs.

        """
        coarse_cutoffs = None
        if args.bin_strategy == BinStrategy.LINEAR:
            coarse_cutoffs = get_coarse_cutoffs(
                args.ncoarse_bins, args.coarse_low, args.coarse_high)
        if coarse_cutoffs is None:
            msg = f"Bin strategy {args.bin_strategy} not implemented for given args: {args}."
            raise RuntimeError(msg)
        coarse_cutoffs = torch.tensor(coarse_cutoffs)
        logger.debug("Obtained coarse cutoffs: %s", coarse_cutoffs)
        return coarse_cutoffs
