"""Utility collate functions for the DataLoaders

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/collate_utils.py

import torch
from sutranets.ml.constants import ExampleKeys
from sutranets.ml.multivariate.tensor_list import TensorList


class CollateUtils():
    """Utility collate functions for the DataLoaders."""

    @staticmethod
    def __cat_lsts(all_keys, all_lsts):
        """Helper to make the collated batch from the collected keys and
        lists, using concatenate.

        """
        collated = {}
        for key, lst in zip(all_keys, all_lsts):
            if lst:
                if isinstance(lst[0], TensorList):
                    batch_tensor = TensorList.batch_up(lst)
                else:
                    batch_tensor = torch.cat(lst, dim=1)
            else:
                batch_tensor = None
            collated[key] = batch_tensor
        return collated

    @classmethod
    def batching_collator(cls, batch):
        """A helper function to use with the DataLoader - this one works with
        batches in higher dimensions.

        """
        all_keys = [ExampleKeys.INPUT, ExampleKeys.TARGET,
                    ExampleKeys.ORIGINALS, ExampleKeys.COARSES,
                    ExampleKeys.FINES, ExampleKeys.FINES2]
        all_lsts = [[] for _ in all_keys]
        all_meta = []
        all_none = True
        for example in batch:
            if example is None:
                continue
            all_none = False
            for key, lst in zip(all_keys, all_lsts):
                data = example.get(key)
                if data is not None:
                    lst.append(data)
            meta = example.get(ExampleKeys.META)
            if meta is not None:
                all_meta += meta
        if all_none:
            return None
        collated = cls.__cat_lsts(all_keys, all_lsts)
        if all_meta:
            collated[ExampleKeys.META] = all_meta
        return collated
