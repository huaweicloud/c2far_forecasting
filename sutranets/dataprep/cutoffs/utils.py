"""Various utilities related to cutoffs.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/dataprep/cutoffs/utils.py
import os
import json
import logging
import torch
logger = logging.getLogger("sutranets.dataprep.cutoffs.utils")
COARSE_CUTOFFS_BASE = "coarse_cutoffs"
EXT = "json"


class CutoffsError(Exception):
    """Base class for exceptions in this module, i.e., for doing
    things related to cutoffs.

    """


class LoadCutoffsError(CutoffsError):
    """Error related to loading the cutoffs."""


def save_one_file(target_dir, file_base, my_tensor):
    """Save the given tensor my_tensor to the file_base in the target_dir.

    """
    if my_tensor is not None:
        full_base = file_base + "." + EXT
        path = os.path.join(target_dir, full_base)
        logger.info("Saving %s to %s", file_base, path)
        with open(path, "w", encoding="utf-8") as ofile:
            json.dump(my_tensor.tolist(), ofile)


def save_cutoffs(target_dir, coarse_cutoffs):
    """Helper to save the coarse cutoffs to the target DIR (if not None).

    """
    if target_dir is not None:
        if coarse_cutoffs is not None:
            save_one_file(target_dir, COARSE_CUTOFFS_BASE, coarse_cutoffs)


def __load_one_file(target_dir, file_base, *, ok_if_missing=False):
    """Load the given tensor my_tensor from the file_base in the
    target_dir.

    Arguments:

    target_dir: String, where to look for the file.

    file_base: String, the part of the file before the extension.

    Optional Args:

    ok_if_missing: Boolean, if True, do not raise an error if the
    file is missing, otherwise, *do* raise such an error.

    Returns:

    torch.Tensor, of shape N: just the floats being loaded (cutoffs),
    or None if missing and ok_if_missing==True.

    """
    full_base = file_base + "." + EXT
    path = os.path.join(target_dir, full_base)
    logger.info("Loading %s from %s", file_base, path)
    if not os.path.exists(path):
        if not ok_if_missing:
            msg = f"Loading {file_base} from {path} but does not exist."
            raise LoadCutoffsError(msg)
        logger.debug("File path %s missing.", path)
        return None
    with open(path, "r", encoding="utf-8") as ofile:
        obj = json.load(ofile)
    return torch.tensor(obj)


def load_cutoffs(saved_dir):
    """Load cutoffs from the saved_dir and return them.

    """
    coarse_cutoffs = __load_one_file(saved_dir, COARSE_CUTOFFS_BASE)
    return coarse_cutoffs
