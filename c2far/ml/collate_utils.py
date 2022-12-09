"""Utility collate functions for the DataLoaders

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/collate_utils.py

import torch
from c2far.ml.constants import ExampleKeys


class CollateUtils():
    """Utility collate functions for the DataLoaders."""

    @staticmethod
    def batching_collator(batch):
        """A helper function to use with the DataLoader - this one works with
        batches in higher dimensions.

        Convention in PyTorch is to have the input dimensions like
        this: SEQ_LEN x NBATCH x NFEATURES - where the batch dimension
        is the middle dimension.

        Arguments:

        batch: an iterable over example (dicts) in a Dataset (because
        it's a collator), each dict with the usual example key/value
        pairs.  Note the inputs/outputs/targets/etc. in the examples
        most likely have a batch size of 1, but we also use this
        function in the batch_predictor to collate samples for
        multi-step-ahead generation, and it works fine there too where
        the input is already in batches - we just use this function to
        collate batches into bigger batches.

        Returns:

        collated, a single Dict in a Dataset, with the usual keys, but
        the values now contain the usual payload, but all stacked
        together.

        """
        all_keys = [ExampleKeys.INPUT, ExampleKeys.TARGET,
                    ExampleKeys.ORIGINALS, ExampleKeys.COARSES]
        all_lsts = [[], [], [], []]
        for example in batch:
            # Assume anything may be absent:
            for key, lst in zip(all_keys, all_lsts):
                data = example.get(key)
                if data is not None:
                    lst.append(data)
        # Cat them together along batch dim to make tensors:
        collated = {}
        for key, lst in zip(all_keys, all_lsts):
            if lst:  # If we collected anything:
                batch_tensor = torch.cat(lst, dim=1)
            else:
                batch_tensor = None
            collated[key] = batch_tensor
        return collated
