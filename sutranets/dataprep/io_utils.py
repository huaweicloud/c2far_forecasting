"""Utilities related to I/O and data preparation: reading, writing
data, etc.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/dataprep/io_utils.py

import linecache
import logging
from sutranets.dataprep.constants import CSV_SEP, SEQ_VAL_SEP
# Logging in this file:
logger = logging.getLogger("sutranets.dataprep.io_utils")
# Constants just for this file:
REPORT = 5000
CUTOFF_SEP = " "


def parse_input_line(line):
    """Helper to parse a single input line and return meta + vals."""
    line = line.rstrip()
    parts = line.split(CSV_SEP)
    valstr = parts[-1]
    vals = [float(v) for v in valstr.split(SEQ_VAL_SEP)]
    return parts[:-1], vals


def get_meta_vals(input_fn, idx):
    """Get the meta AND values from the given input file."""
    # Linecache is indexed from 1, so adjust here:
    line = linecache.getline(input_fn, idx + 1)
    meta, vals = parse_input_line(line)
    return meta, vals
