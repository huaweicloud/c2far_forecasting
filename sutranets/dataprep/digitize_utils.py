"""Single location for our utilities in terms of turning a sequence of
real values into a sequence of binned values, and vice versa.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/dataprep/digitize_utils.py
import torch


class DigitizeUtils():
    """Static methods of this class let us encode and decode sequences to
    and from values.

    """
    @classmethod
    def torch_digitize_normed(cls, normed_values, val_cutoffs):
        """Given the conditioning window and the originals [tensor to be
        encoded], get the encoded values.

        Input:

        normed_values: torch.tensor, the originals, but already
        normalized by the conditioning window.

        val_cutoffs: torch.tensor

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
        # For increasing bins, do it so that bins[i-1] <= x <
        # bins[i]. Note to get the same behaviour as the numpy
        # encode_values, the 'right' argument is flipped for bucketize
        # (it's more like numpy's searchsorted):
        indices = torch.bucketize(values, cutoffs, right=True)
        # See: https://github.com/pytorch/pytorch/issues/43458
        return indices

    @staticmethod
    def torch_prepare_cutoffs(cutoffs_arr, *, fixed_lower, fixed_upper):
        """Preprocessing step to prepare lower/upper cutoffs.

        Returns:

        lower_cutoffs, upper_cutoffs, Tensors[Float], the cutoffs

        """
        if fixed_lower is None or fixed_upper is None:
            raise RuntimeError("Must pass fixed lower/upper values for cutoffs")
        if fixed_upper <= cutoffs_arr[-1]:
            msg = f"Fixed upper {fixed_upper} not beyond end of cutoffs {cutoffs_arr}"
            raise RuntimeError(msg)
        if fixed_lower >= cutoffs_arr[0]:
            msg = f"Fixed lower {fixed_lower} not beyond start of cutoffs {cutoffs_arr}"
            raise RuntimeError(msg)
        append_val = torch.tensor([fixed_upper], device=cutoffs_arr.device)
        upper_cutoffs = torch.cat([cutoffs_arr, append_val], dim=0)
        prepend_val = torch.tensor([fixed_lower], device=cutoffs_arr.device)
        lower_cutoffs = torch.cat([prepend_val, cutoffs_arr], dim=0)
        return lower_cutoffs, upper_cutoffs

    @staticmethod
    def torch_decode_values(encoded_arr, lower_cutoffs, upper_cutoffs):
        """Decode from cutoffs+bings to values.

        encoded: tensor: NVALUES (e.g. NSEQ*NBATCH in flattened mode
        of dimension 1): a list of all the encoded values.

        lower_cutoffs: tensor: NCUTOFFS - the lower cutoff points for
        decoding.

        upper_cutoffs: tensor: NCUTOFFS - the upper cutoff points for
        decoding.

        """
        upper_bounds = upper_cutoffs[encoded_arr]
        lower_bounds = lower_cutoffs[encoded_arr]
        values = torch.rand(lower_bounds.shape,
                            device=encoded_arr.device)
        values = values * (upper_bounds - lower_bounds) + lower_bounds
        return values
