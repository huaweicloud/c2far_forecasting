"""ABC for a guesser that turns outputs, originals, and values into
guesses that we can compare to originals, or use to compute some loss
with respect to the originals.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/loss_functions/guesser.py
from abc import ABC, abstractmethod


class Guesser(ABC):
    """Defines the interface for classes that will return guesses.

    """
    def __init__(self, strategy):
        self.strategy = strategy

    @abstractmethod
    def get_guesses(self, outputs, originals, values):
        pass

    def get_info(self):
        _info = f"(Unmap:strat={self.strategy.value})"
        return _info
