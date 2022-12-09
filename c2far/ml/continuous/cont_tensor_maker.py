"""A class with the job of making tensors for the input and output,
but where those inputs and outputs are real-valued (continuous).

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/continous/cont_tensor_maker.py

from c2far.ml.constants import INPUT_DTYPE


class ContTensorMaker():
    """Makes tensors for inputs and outputs.  Here, everything is based on
    the *normalized* real-valued originals.

    """
    __DUMMY_VAL = 0.0

    def __init__(self, genmode):
        """Get the tensor_maker initialized.

        Args:

        genmode: Boolean, whether we use all the input sequences for
        generation (True), or we use the last element as a target
        (False).  Default: False.

        Also returns the input (1) and output (2) dimensionality.  2
        outputs for mean/variance of a Gaussian.

        """
        self.ninput = 1
        self.noutput = 2
        self.genmode = genmode

    def get_ninput(self):
        """Getter for the ninput

        """
        return self.ninput

    def get_noutput(self):
        """Getter for the noutput

        """
        return self.noutput

    def encode_input(self, normed):
        """Given the normed input data, we encode it.

        Arguments:

        normed: Tensor(N+1)

        Output:

        Tensor(N-1 x NBATCH=1 x 1) (or N x NBATCH x 1 in genmode=True)

        """
        normed = normed[1:]
        if not self.genmode:
            normed = normed[:-1]
        normed = normed.clone()
        # The input features are basically just normed, but what if
        # normed have nan because series goes beyond end_pt_s? Fix it:
        normed[normed.isnan()] = self.__DUMMY_VAL
        return normed.reshape(-1, 1, 1).to(INPUT_DTYPE)

    @staticmethod
    def encode_originals(originals):
        """Converts the flat originals tensor into a 3D tensor of the kind of
        dimensionality we've come to expect, and which we can later
        collate into a batch easily:

        Arguments:

        originals: Tensor: NSEQ+2

        Returns

        Tensors: NSEQ+2 x NBATCH=1 x 1

        """
        # Just make it a tensor, then add the two extra dims:
        return originals.unsqueeze(1).unsqueeze(1)

    def encode_target(self, normed):
        """Given the key normed output data, we encode it.

        It's very similar to encode_input, except we use a different
        part of normed (skipping first value).

        Arguments:

        normed: Tensor(N)

        Output:

        Tensor(N-1 x NBATCH=1 x 1)

        """
        normed = normed[2:].clone()
        # And again, don't want any nans creeping in here (they are
        # used to signal points beyond our data, and seem to mess up
        # gradients even when not involved in loss!)
        normed[normed.isnan()] = self.__DUMMY_VAL
        return normed.reshape(-1, 1, 1)
