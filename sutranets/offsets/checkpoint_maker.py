"""Given a set of offsets, the checkpoint maker will divide the
offsets into subsets (checkpoints), and iterate through these subsets
as requested.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/offsets/checkpoint_maker.py
import logging
from random import randint
from sutranets.dataprep.get_offsets import OFFSET_SEP
logger = logging.getLogger("sutranets.ml.offsets.checkpoint_maker")


class CheckpointMaker():
    """Makes all the offsets up until the next checkpoint.  Reads the
    offsets_fn, but internally, it avoids reading them all into memory
    at once.

    """
    def __init__(self, offsets_fn, *, ncheckpoint=-1,
                 randomize_start=False):
        self.offsets_fn = offsets_fn
        with open(self.offsets_fn, encoding="utf-8") as offsets_file:
            self.noffsets = sum(1 for _ in offsets_file)
        if ncheckpoint is None:
            raise RuntimeError("ncheckpoint == None is no longer supported.")
        if ncheckpoint < 0:
            self.ncheckpoint = self.noffsets
        else:
            self.ncheckpoint = ncheckpoint
        self.curr_off = 0
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
        """Gather nlines lines from the file, or stop at end.

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
            nremaining = self.ncheckpoint - len(lines)
            while nremaining > 0:
                self.file_off = 0
                offsets_file.seek(0)
                nbytes = self.__add_nlines(lines, offsets_file, nremaining)
                nremaining = self.ncheckpoint - len(lines)
        self.file_off += nbytes
        self.curr_off = (self.curr_off + self.ncheckpoint) % self.noffsets
        return lines

    def yield_checkpoints(self):
        """Iterator to yield all the offsets for the next checkpoint."""
        while True:
            cp_offsets = []
            lines = self.__grab_checkpoint_lines()
            for line in lines:
                offset_parts = [int(p) for p in line.split(OFFSET_SEP)]
                cp_offsets.append(offset_parts)
            yield cp_offsets
