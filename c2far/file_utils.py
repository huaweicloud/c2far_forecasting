"""General functions related to files and file I/O.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/file_utils.py

import argparse
from datetime import datetime
import os
import pathlib
import uuid

DATE_STAMP_FORMAT = "%Y%m%d_%H%M_%S"
LOG_SHORT_FN = "log.txt"


def get_run_date():
    """Move logic for getting a run_date string here, so it can be used in
    different programs.

    """
    # Add a random part so that there are no conflicts if we launch at
    # exactly the same time:
    my_uuid = str(uuid.uuid4())[0:6]
    return (datetime.strftime(datetime.now(), DATE_STAMP_FORMAT) +
            "." + my_uuid)


def setup_outdir(out_dir):
    """If it's given, check the output dir exists, and if not, make it,
    and then do everything in a timestamped subdir

    If None is given, return None likewise.

    """
    if out_dir is not None:
        if not os.path.exists(out_dir):
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        sim_run_date = get_run_date()
        target_dir = os.path.join(out_dir, sim_run_date)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        return target_dir
    return None


class MyArgumentTypes():
    """Some types that are used in checking command-line arguments."""

    @staticmethod
    def filenametype(string):
        """Type for checking input files exist, without opening them right
        away."""
        if not os.path.exists(string):
            raise argparse.ArgumentTypeError(
                f"File {string} does not exist.")
        return string

    @staticmethod
    def outdirtype(string):
        """Type for checking output directory, without opening them right
        away.

        """
        mydir = os.path.dirname(string)
        if not mydir == '':
            if not os.path.isdir(mydir):
                raise argparse.ArgumentTypeError(
                    f"Output directory {string} does not exist.")
        return string
