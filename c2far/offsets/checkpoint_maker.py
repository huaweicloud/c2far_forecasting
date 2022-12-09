"""Given a set of offsets, the checkpoint maker will divide the
offsets into subsets (checkpoints), and iterate through these subsets
as requested.  This module has minimal knowledge of what an of an
offset "is".  It mainly just works with entire lines in the offsets
file, although it does split them at the yield phase using the
OFFSET_SEP.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/offsets/checkpoint_maker.py
import logging
from random import randint
OFFSET_SEP = "\t"
logger = logging.getLogger("c2far.ml.offsets.checkpoint_maker")


class CheckpointMaker():
    """Makes all the offsets up until the next checkpoint.  Reads the
    offsets_fn, but internally, it avoids reading them all into memory
    at once (since we just need one consecutive subset each time
    anyways).

    """
    def __init__(self, offsets_fn, *, ncheckpoint=-1,
                 randomize_start=False):
        """Initialize the checkpoint maker given all the offsets.  Initialize
        to return offsets up until the next checkpoint.

        Arguments:

        offsets_fn: String, the file containing all the offsets into
        the time series data.

        ncheckpoint: Int, how many examples to process before
        exhausting the getitem iterator (allows us to check progress).
        Pass '-1' to default to all the offsets.

        randomize_start: Boolean, whether to randomize our starting
        position in the offsets.  No reason not to do this each time,
        really, but I keep it False by default so that UTs work
        without needing changes.

        naugments: Int, how many data augmentations we have of each
        time series.  Default==2: we have vcpus and memory versions of
        each.

        """
        self.offsets_fn = offsets_fn
        with open(self.offsets_fn, encoding="utf-8") as offsets_file:
            self.noffsets = sum(1 for _ in offsets_file)
        if ncheckpoint is None:
            # help with this obsolete flag:
            raise RuntimeError("ncheckpoint == None is no longer supported.")
        if ncheckpoint < 0:
            # We have one flag: -1 means use all:
            self.ncheckpoint = self.noffsets
        else:
            self.ncheckpoint = ncheckpoint
        # Internally, we'll just monitor which offset we are on:
        self.curr_off = 0
        # And where to seek in file for that offset:
        self.file_off = 0
        if randomize_start:
            nskip = randint(0, self.noffsets - 1)
            msg = f"Skipping first {nskip} offs in " \
                  f"{self.offsets_fn} to randomize_start"
            logger.info(msg)
            self.curr_off = nskip
            with open(self.offsets_fn, encoding="utf-8") as offsets_file:
                self.file_off = self.__get_file_offset(offsets_file)

    def __get_file_offset(self, offsets_file):
        """Determine where to seek to in a file for the curr_off."""
        file_offset = 0
        for file_idx, line in enumerate(offsets_file):
            if file_idx == self.curr_off:
                return file_offset
            file_offset += len(line)
        msg = f"Offset {self.curr_off} beyond end of file"
        raise RuntimeError(msg)

    @staticmethod
    def __add_nlines(lines, offsets_file, tot_lines):
        """Helper to gather nlines lines from the file, or stop at end.
        Records how many bytes we've read (so we can advance the file
        offset later by this amount, for next time).

        """
        tot_bytes = 0
        for nlines, line in enumerate(offsets_file):
            if nlines >= tot_lines:
                break
            lines.append(line)
            tot_bytes += len(line)
        return tot_bytes

    def __grab_checkpoint_lines(self):
        """Efficiently grab the next ncheckpoint lines from the offset_fn."""
        lines = []
        with open(self.offsets_fn, encoding="utf-8") as offsets_file:
            offsets_file.seek(self.file_off)
            nbytes = self.__add_nlines(lines, offsets_file, self.ncheckpoint)
            # Are there any remaining?  Did we hit end of file first?
            nremaining = self.ncheckpoint - len(lines)
            while nremaining > 0:
                self.file_off = 0
                offsets_file.seek(0)
                nbytes = self.__add_nlines(lines, offsets_file, nremaining)
                nremaining = self.ncheckpoint - len(lines)
        self.file_off += nbytes
        # Start here next time:
        self.curr_off = (self.curr_off + self.ncheckpoint) % self.noffsets
        return lines

    def yield_checkpoints(self):
        """Iterator to yield all the offsets for the next checkpoint."""
        while True:
            cp_offsets = []
            lines = self.__grab_checkpoint_lines()
            for line in lines:
                offset_parts = [int(p) for p in
                                line.split(OFFSET_SEP)]
                cp_offsets.append(offset_parts)
            yield cp_offsets
