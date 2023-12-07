"""Helper to dynamically create the coarse cutoffs.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/dataprep/cutoffs/create_cutoffs.py
import logging
import torch
from sutranets.dataprep.cutoffs.linear_cutoffs import get_coarse_cutoffs
from sutranets.dataprep.cutoffs.utils import load_cutoffs
from sutranets.ml.constants import BinStrategy

logger = logging.getLogger("sutranets.dataprep.cutoffs.create_cutoffs")


class CreateCutoffs():
    """Utilities for creating cutoffs."""

    @classmethod
    def __get_coarse_cutoffs(cls, args, force_linear):
        """Helper with code moved out of do_create_cutoffs purely because that
        code was getting too complex.

        Arguments:

        args: NamedTuple CLI args, including, e.g., ncoarse_bins.

        force_linear: Boolean, if True, do linear cutoffs regardless
        of what's in the args, else follow args.

        Returns:

        coarse_cutoffs: Tensor[NBINS], list of coarse cutoffs.

        """
        coarse_cutoffs = None
        if force_linear or args.bin_strategy == BinStrategy.LINEAR:
            coarse_cutoffs = get_coarse_cutoffs(
                args.ncoarse_bins, args.coarse_low, args.coarse_high)
        coarse_cutoffs = torch.tensor(coarse_cutoffs)
        logger.debug("Obtained coarse cutoffs: %s", coarse_cutoffs)
        return coarse_cutoffs

    @classmethod
    def do_create_cutoffs(cls, args, force_linear=False):
        """Entry point to create coarse cutoffs based on the given arguments.

        Arguments:

        force_linear: Boolean, if True, do linear cutoffs regardless
        of what's in the args, else follow args.

        Returns:

        coarse_cutoffs: Tensor[NBINS], list of coarse cutoffs.

        """
        coarse_cutoffs = cls.__get_coarse_cutoffs(args, force_linear)
        return coarse_cutoffs

    @staticmethod
    def __validate_cutoffs(coarse_cutoffs, args):
        """Validate the created info matches what we expect from the args."""
        ncuts = len(coarse_cutoffs)
        nbins = args.ncoarse_bins
        if ncuts != nbins - 1:
            msg = f"Mismatch in {ncuts} cutoffs and {nbins} bins"
            raise RuntimeError(msg)

    @classmethod
    def get_cutoffs(cls, args):
        """Load the cutoffs from the saved_dir, if they are in that dir, or
        make them if they are not, and return them.

        Arguments:

        args: NamedTuple CLI args, including, e.g., ncoarse_bins,
        nfine_bins, dataset info, etc.

        Returns:

        coarse_cutoffs: Tuple, of 2 tensors, of length NBINS-1,
        NBINS-1

        """
        if args.cutoff_dir is not None:
            coarse_cutoffs = load_cutoffs(args.cutoff_dir)
        else:
            coarse_cutoffs = cls.do_create_cutoffs(args)
        cls.__validate_cutoffs(coarse_cutoffs, args)
        return coarse_cutoffs


def setup_cutoffs(cutoffs, args):
    """Read or create the coarse cutoffs, or return None.

    Arguments:

    cutoffs: coarse_cuts, or None if not passing directly.

    args: standard CLI args.

    Returns:

    coarse_cuts

    """
    if args.ncoarse_bins is None:
        raise RuntimeError("Need args.ncoarse_bins.")
    if args.coarse_high is None or args.coarse_low is None:
        raise RuntimeError("Need coarse high/low for binning")
    if cutoffs is None:
        coarse_cutoffs = CreateCutoffs.get_cutoffs(args)
    else:
        coarse_cutoffs = cutoffs
    return coarse_cutoffs
