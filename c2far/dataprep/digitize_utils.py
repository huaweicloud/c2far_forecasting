"""Utilities for turning a sequence of real values into a sequence of
binned elements.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/dataprep/digitize_utils.py

import logging
import torch
logger = logging.getLogger("c2far.dataprep.digitize_utils")


class DigitizeUtils():
    """Static methods of this class let us encode and decode sequences to
    and from real values.

    """
    @classmethod
    def torch_digitize_normed(cls, normed_values, val_cutoffs):
        """Given the normed tensor to be encoded, get the encoded values.

        Input:

        normed_values: torch.tensor, the originals, but already
        normalized by the conditioning window.

        val_cutoffs: torch.tensor, the cutoffs for binning.

        Returns:

        encoded_values: torch.tensor, the N encoded values

        """
        encoded_values = cls.torch_encode_values(
            normed_values, val_cutoffs)
        return encoded_values

    @staticmethod
    def torch_encode_values(values, cutoffs):
        """Encode the given list of values using the cutoffs.

        """
        # As above (in encode_values), for increasing bins, do it so
        # that bins[i-1] <= x < bins[i]. Note to get the same
        # behaviour as the encode_values using numpy, the 'right'
        # argument is flipped for bucketize (it's more like numpy's
        # searchsorted):
        indices = torch.bucketize(values, cutoffs, right=True)
        return indices

    @staticmethod
    def torch_prepare_cutoffs(cutoffs_arr, *, fixed_lower, fixed_upper):
        """Rather than preparing the lower/upper cutoffs each time in
        torch_decode_values, we do it just once, here, as a
        preprocessing step.

        Recall, encoding i means bins[i-1] <= x < bins[i]. But what
        about the first/last bin?  We require you pass fixed_upper and
        fixed_lower for this purpose.

        Returns:

        lower_cutoffs, upper_cutoffs, Tensors[Float], the cutoffs

        """
        # Append/prepend the fixed_upper/fixed_lower to give us the lower/uppers:
        append_val = torch.tensor([fixed_upper], device=cutoffs_arr.device)
        upper_cutoffs = torch.cat([cutoffs_arr, append_val], dim=0)
        prepend_val = torch.tensor([fixed_lower], device=cutoffs_arr.device)
        lower_cutoffs = torch.cat([prepend_val, cutoffs_arr], dim=0)
        return lower_cutoffs, upper_cutoffs

    @staticmethod
    def torch_decode_values(encoded_arr, lower_cutoffs, upper_cutoffs,
                            *, special_value_decode=True):
        """Uniform decoding from encoded values and cutoffs to output values.

        encoded: tensor: NVALUES (e.g. NSEQ*NBATCH in flattened mode
        of dimension 1): a list of all the encoded values.

        lower_cutoffs: tensor: NCUTOFFS - the lower cutoff points for
        decoding.

        upper_cutoffs: tensor: NCUTOFFS - the upper cutoff points for
        decoding.

        Optional arguments:

        special_value_decode: Boolean: if True, then always output 0
        or 1 if bin contains such a value. This is crucial for not
        getting 200% error on SMAPE when we're off any amount.

        """
        upper_bounds = upper_cutoffs[encoded_arr]
        lower_bounds = lower_cutoffs[encoded_arr]
        values = torch.rand(lower_bounds.shape,
                            device=encoded_arr.device)
        values = values * (upper_bounds - lower_bounds) + lower_bounds
        if special_value_decode:
            for spec_val in [0, 1]:
                matches = (lower_bounds <= spec_val) & (spec_val < upper_bounds)
                values[matches] = spec_val
        return values
