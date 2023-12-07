"""Determines the input and output sizes for tensors made by
BinTensorMaker.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/bin_tensor_maker_dims.py


class BinTensorMakerDims():
    """Friend class of BinTensorMaker that can compute the input and
    output dimensionality of the tensors that BinTensorMaker makes.

    """
    def __init__(self, coarse_cutoffs, nfine_bins, nfine2_bins, *,
                 extremas, lagfeat_period, nmultiv_siblings=None):
        """Using standard BinTensorMaker info, compute the in/out dimensionality."""
        ncoarse_bins = len(coarse_cutoffs) + 1
        nbin_covariates = 0
        if lagfeat_period is not None and lagfeat_period:
            nbin_covariates += 1
        if nmultiv_siblings is not None:
            nbin_covariates += nmultiv_siblings
        self.ninput, self.noutput = self.__get_in_out_dims(
            ncoarse_bins, nfine_bins,
            nfine2_bins, extremas, nbin_covariates)

    @staticmethod
    def __get_in_out_dims(ncoarse_bins, nfine_bins,
                          nfine2_bins, extremas, nbin_covariates):
        """Helper to get the ninput and noutput sizes, depending on which
        features we are using.

        """
        ninput = ncoarse_bins
        nbinned_output = ncoarse_bins
        if nfine_bins is not None:
            ninput += (nfine_bins + ncoarse_bins)
            nbinned_output += nfine_bins
        if nfine2_bins is not None:
            if nfine_bins is None:
                raise RuntimeError("Can only use fine2 if using fine.")
            ninput += (nfine2_bins + nfine_bins)
            nbinned_output += nfine2_bins
        ninput += nbinned_output * nbin_covariates
        noutput = nbinned_output
        if extremas:
            noutput += 2
            ninput += 1
        return ninput, noutput

    def get_ninput(self):
        return self.ninput

    def get_noutput(self):
        return self.noutput
