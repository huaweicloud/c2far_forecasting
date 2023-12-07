"""Container to hold lists of tensors, intended for multivariate
INPUTS, ORIGINALS, COARSES, etc., so we can store these for all series
in the multivariate group, and apply similar operations to them as we
do on individual tensors.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/multivariate/tensor_list.py
import torch


class TensorList():
    def __init__(self, init_lst=None):
        if init_lst is None:
            self.lst_of_tensors = []
            self.shape = None
        else:
            self.lst_of_tensors = init_lst
            self.shape = init_lst[0].shape

    def append(self, tens):
        """Add the given tensor to the list."""
        if not self.lst_of_tensors:
            self.shape = tens.shape
        self.lst_of_tensors.append(tens)

    def to(self, device):
        """Move all tensors to the device.

        """
        new_lst = []
        for tens in self.lst_of_tensors:
            new_tens = tens.to(device)
            new_lst.append(new_tens)
        self.lst_of_tensors = new_lst
        return self

    def __getitem__(self, idx):
        """Return the item at the given index, this enables use of []
        operator.

        """
        return self.lst_of_tensors[idx]

    def __len__(self):
        """Return the length of the list at present."""
        return len(self.lst_of_tensors)

    def get_lst_of_tensors(self):
        """Return the list of tensors."""
        return self.lst_of_tensors

    @classmethod
    def batch_up(cls, lst_of_tlists):
        """Return a new TensorList, that contains the batching of the tensors
        in the passed lst of TensorList.  Basically tensors at the
        same index of each TensorList will get batched.

        """
        outer_lists = []
        for tlist in lst_of_tlists:
            inner_list = tlist.get_lst_of_tensors()
            outer_lists.append(inner_list)
        batched_tlst = []
        for level_list in zip(*outer_lists):
            level_concat = torch.cat(level_list, dim=1)
            batched_tlst.append(level_concat)
        return cls(batched_tlst)
