"""ABC for a guesser that turns outputs, originals, and values into
guesses that we can compare to originals, or use to compute some loss
with respect to the originals.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/loss_functions/guesser.py

from abc import ABC, abstractmethod


class Guesser(ABC):
    """Defines the interface for classes that will return guesses.

    """
    def __init__(self, strategy):
        """All guessers need some kind of strategy"""
        self.strategy = strategy

    @abstractmethod
    def get_guesses(self, outputs, originals, values):
        """Generic code that converts our outputs to guesses.

        Arguments:

        outputs: Tensor: NSEQ x NBATCH x NBINS, typically the logits
        for each bin, for each element of each sequence of each
        tensor.

        originals: Tensor: NSEQ+2 x NBATCH x 1, the original values
        for each batch of sequences.

        values: Tensor: NSEQ+2 x NBATCH x 1 (or None), the encoded
        values for each batch of sequences.  Can pass None if not
        using certain intra-bin-decoding methods.

        """

    def get_info(self):
        """Return some information about the guesser that can be used in
        reporting or logging.

        """
        _info = f"(Unmap:strat={self.strategy.value})"
        return _info
