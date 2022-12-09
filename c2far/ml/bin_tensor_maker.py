"""A class with the job of making tensors for the input and output,
for the binned representations.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/ml/bin_tensor_maker.py

import torch
from c2far.ml.constants import INPUT_DTYPE
from c2far.ml.fc_digitizer import FCDigitizer
from c2far.ml.loss_functions.constants import IGNORE_INDEX
DUMMY_BIN = 0


class BinTensorMaker():
    """Make tensors for inputs and outputs, as requested.

    By 'prev covar', we mean the (previous) covariates, e.g. there may
    also be a single-dimensional (previous) 'normed' value if doing
    extreme bins.

    if NO fine bins:

    - input is:  prev coars, prev covar

    - output is: curr coars

    if fine bins:

    - *coarse* input is: prev coars, prev fine_, prev covar
    - *fine* input is:               prev fine_, prev covar, curr coars
    - *total* input is:  prev coars, prev fine_, prev covar, curr coars

    - output is:         curr coars, curr fine_

    if fine2 bins:

    - *coarse* input is: prev coars, prev fine_, prev fine2, prev covar
    - *fine* input is:               prev fine_, prev fine2, prev covar, curr coars
    - *fine2* input is:                          prev fine2, prev covar, curr coars, curr fine_
    - *total* input is:  prev coars, prev fine_, prev fine2, prev covar, curr coars, curr fine_

    - output is: curr coars, curr fine_, curr fine2

    Update: if delta_cutoffs is None, then we don't have these delta
    features.

    """
    # This will not actually get read, it's purely to put in something
    # reasonable for the ignored ones:
    __DUMMY_VAL = 0.0

    def __init__(self, coarse_cutoffs, coarse_low, coarse_high,
                 nfine_bins, nfine2_bins, genmode, *, extremas):
        """Get the tensor_maker initialized with the mappings that it needs to
        do its thing.

        Args:

        coarse_cutoffs: Tensor[NBINS], list of value cutoffs.

        coarse_low: Float, the lowest implicit boundary for the coarse
        cutoffs.

        coarse_high: Float, the highest implicit boundary for the
        coarse cutoffs.

        nfine_bins: Int, how many fine-grained bins to use - both for
        input, and for targets, or None not to use them for either.

        nfine2_bins: Int, how many fine2-grained bins to use - both for
        input, and for targets, or None not to use them for either.

        genmode: Boolean, whether we use all the input sequences for
        generation (True), or we use the last element as a target
        (False).  Default: False.

        Keyword-only args:

        extremas: Boolean, if True, we also include outputs for the
        extreme bins (e.g., the alpha in a Pareto distribution).

        """
        self.extremas = extremas
        if coarse_cutoffs is None:
            raise RuntimeError("coarse_cutoffs required for bin tensor maker.")
        self.nfine_bins = nfine_bins
        self.nfine2_bins = nfine2_bins
        # We pass extremas=True to the digitizer so that it returns the
        # previous normed value, which then, as not None, gets encoded below:
        self.fc_digitizer = FCDigitizer(coarse_cutoffs, coarse_low,
                                        coarse_high, nfine_bins,
                                        nfine2_bins, extremas=extremas)
        self.ncoarse_bins = len(coarse_cutoffs) + 1
        self.ninput = self.ncoarse_bins
        self.noutput = self.ncoarse_bins
        if self.nfine_bins is not None:
            self.ninput += (self.nfine_bins + self.ncoarse_bins)
            self.noutput += self.nfine_bins
        if self.nfine2_bins is not None:
            if self.nfine_bins is None:
                raise RuntimeError("Can only use fine2 if using fine.")
            self.ninput += (self.nfine2_bins + self.nfine_bins)
            self.noutput += self.nfine2_bins
        if self.extremas:
            # Output params for low/high extrema, take in normed as feat:
            self.noutput += 2
            self.ninput += 1
        self.genmode = genmode

    def get_ninput(self):
        """Getter for the ninput

        """
        return self.ninput

    @staticmethod
    def compute_nsubsets(ninput, ncoarse_bins, nfine_bins,
                         nfine2_bins, extremas):
        """Return how many features in the coarse, fine, and fine2 (input)
        subsets.  Actually, the first three of these will be equal
        (see sliding window above), but their number depends on
        whether using fines (nfine is not None) or fines2 (nfine2 is
        not None).  If extremas is True, we use one of those (even
        though we have two extremas outputs).

        Arguments:

        ncoarse_bins: Int, how many coarse bins there are.

        nfine_bins: Int, how many fine bins there are, or None if not using.

        nfine2_bins: Int, how many fine2 bins there are, or None if not using.

        extremas: Boolean, whether to also give the number of extrema features.

        Returns:

        ncoarse_feats: Int, how many features in the coarse subsets,
        can be None if nfine/nfine2 are also None (then it's irrelevant).

        nfine_feats: Int, how many features in the fine subsets

        nfine2_feats: Int, how many features in the fine2 subsets

        nextrema_feats: Int, how many features in the coarse subsets

        """
        nextrema = 1 if extremas else None
        # If normed are missing, ninput will be smaller, so logic
        # works either way:
        if nfine_bins is None:
            # If only coarse, all features are coarse:
            return ninput, None, None, nextrema
        if nfine2_bins is None:
            # All but current coarse (for coarse prediction) / prev
            # coarse (for fine prediction), respectively:
            nfeats = ninput - ncoarse_bins
            return nfeats, nfeats, None, nextrema
        # All but current/prev coarse and current/prev fine:
        nfeats = ninput - ncoarse_bins - nfine_bins
        return nfeats, nfeats, nfeats, nextrema

    @staticmethod
    def compute_joint_ninput(ncoarse_subset, ncoarse):
        """The BinTensorMaker knows how many total inputs we would have if
        there are ncoarse_subset in the coarse part, and ncoarse bins
        overall, and we are doing joint coarse+fine only.

        We use this helper to create a JointLSTM from a path, where we
        can only red the ninput for one of the LSTMs.

        Works whether extrema features or not.

        """
        # Hint: this is how many (+ current coarse for the fine LSTM).
        # It works when yes/no for extremas because they are already
        # in the coarse subset:
        return ncoarse_subset + ncoarse

    @staticmethod
    def compute_triple_ninput(ncoarse_subset, ncoarse, nfine):
        """The BinTensorMaker knows how many total inputs we would have if
        there are ncoarse_subset in the coarse part, and ncoarse
        coarse bins and nfine fine bins overall, and we are doing
        triple coarse+fine+fine2.

        We use this helper to create a TripleLSTM from a path, where
        we can only red the ninput for one of the LSTMs.

        Works whether extrema features or not.

        """
        # Hint: this is how many (+ current coarse + current fine for
        # the fine2 LSTM):
        return ncoarse_subset + ncoarse + nfine

    def get_noutput(self):
        """Getter for the noutput

        """
        return self.noutput

    @staticmethod
    def __one_hot_values(values, nbins):
        """1-hot-encode the given values, assuming the given number of bins.

        Arguments:

        values: Tensor[NSEQ], the values to encode

        nbins: Int, how many classes in todal.

        Returns:

        Tensor[NSEQ x 1 x NBINS], the values 1-hot-encoded, with an
        extra batch dimension added to allow future collation of
        examples into batches.  This tensor is ready to be
        concatenated with others in the feature dimension.

        """
        if not torch.is_tensor(values):
            # pylint: disable=not-callable
            values = torch.tensor(values)
        tensor = torch.nn.functional.one_hot(values,
                                             num_classes=nbins) \
                                    .to(INPUT_DTYPE) \
                                    .unsqueeze(dim=1)
        return tensor

    def digitize(self, origs, csize, nsize):
        """Return the digitized coarses, fine-grained bins (if using), and
        fine2 (if using), and normed originals.

        """
        return self.fc_digitizer(origs, csize, nsize)

    def __shift_first_three(self, coarses, normed):
        """Helper to shift the coarses, and normed, introduced purely
        because pylint was saying __shift_inputs was too complex.  See
        __shift_inputs for full details.

        Arguments:

        coarses: Tensor(NSEQ + 1)

        normed: Tensor(NSEQ + 1), or None

        Returns:

        Shifted tensors of the above.

        """
        # Strip off "consumed" first coarse/normed:
        pcoarses = coarses[1:]
        pnormed = normed
        if pnormed is not None:
            pnormed = pnormed[1:]
        if not self.genmode:
            # Save last one to be a target
            pcoarses = pcoarses[:-1]
            if pnormed is not None:
                pnormed = pnormed[:-1]
        return pcoarses, pnormed

    def __shift_inputs(self, coarses, fines, fines2, normed):
        """The input here are tensors of indexes, representing ranges of the
        time series.  Depending on what mode we're in, and whether
        joint, adjust what range we can see for each given tensor, and
        return the adjusted (shifted) versions.

        Arguments:

        coarses: Tensor(NSEQ + 1)

        fines: Tensor(NSEQ + 1)

        fines2: Tensor(NSEQ + 1)

        normed: Tensor(NSEQ + 1), or None

        Returns:

        pcoarses: Tensor[]

        pnormed: Tensor[], or None if not using.

        pfines: Tensor[], or None if not doing fine bins

        ccoarses: Tensor[], or None if not doing fine bins

        pfines2: Tensor[], or None if not doing fine2 bins

        cfines: Tensor[], or None if not doing fine2 bins

        Values/lengths of these output tensors depends on following 4 scenarios:

        |    |   |          In genmode?          |                            | What we call these:
        |----|---|:-----------------------------:|:--------------------------:|
        |    |   | N                             | Y                          |
        Joint| N | Save last element:            | Use last element:          |
        targ-|   | covars[:-1]                   | covars[:]                  |  covars
        ets  |   | coarses[1:-1]                 | coarses[1:]                |  pcoarses
        |    | 1 | Fine also uses CURRENT coarse | Use last+extra delta/value |
        |    |   | + fines[1:-1]                 | fines[1:]                  |  pfines
        |    |   | + coarses[2:]                 | coarses[2:] + [DUMMY]      |  ccoarses
        |    | 2 | Fine2 also uses CURRENT fine  | Use last+extra delta/value |
        |    |   | + fines2[1:-1]                | fines2[1:]                 |  pfines2
        |    |   | + fines[2:]                   | fines[2:] + [DUMMY]        |  cfines
        |Extrema?| normed[1:-1]                  | normed[1:]                 |  pnormed

        Joint targets: N: coarse only, 1: coares+fine only, 2:
        corase+fine+fine2.  The items in 2 build on those in 1, which
        build on those in N.

        Note: In genmode with joint targs (1,Y/2,Y), we leverage all
        coarses/fines, but unlike training (genmode=N), we don't have
        very last "current" coarse (and fine for fine2), as we haven't
        generated it (them) yet. In generation, we will sequentially
        first generate coarse, then adjust the feature vector to
        include it as a ccoarse, then generate fine (similarly again
        to generate and encode cfine for fines2 prediction).  But when
        we generate coarse (fine for fine2) here, we need to stick
        something into unused ccoarse (cfines) part of feature vector,
        so we put DUMMY, since we'll overwrite it anyways.

        """
        pcoarses, pnormed = self.__shift_first_three(coarses, normed)
        if self.nfine_bins is None:
            return pcoarses, pnormed, None, None, None, None
        pfines = fines[1:]
        ccoarses = coarses[2:]
        if self.genmode:
            # Need one extra ccoarse, as described above:
            dummy_tens = torch.tensor([DUMMY_BIN], device=ccoarses.device)
            ccoarses = torch.cat([ccoarses, dummy_tens])
        else:
            pfines = pfines[:-1]
        if self.nfine2_bins is None:
            return pcoarses, pnormed, pfines, ccoarses, None, None
        pfines2 = fines2[1:]
        cfines = fines[2:]
        if self.genmode:
            # Need one extra cfine, as described above:
            dummy_tens = torch.tensor([DUMMY_BIN], device=cfines.device)
            cfines = torch.cat([cfines, dummy_tens])
        else:
            pfines2 = pfines2[:-1]
        return pcoarses, pnormed, pfines, ccoarses, pfines2, cfines

    @classmethod
    def __make_normed_input(cls, normed):
        """Helper to make the normed values for features, when doing extrema.

        Arguments:

        values: Tensor[NSEQ], the normed values encode.

        Returns:

        Tensor[NSEQ x NBATCH=1 x NFEATS=1], the single-feature normed
        inputs for every element of the sequence, with an extra batch
        dimension added to allow future collation of examples into
        batches.  This tensor is ready to be concatenated with others
        in the feature dimension.

        """
        # recall NaNs signal points beyond end of our data.  As in
        # cont_tensor_maker (see further rationale there) we prevent them from
        # creeping into our vectors, even when not computing loss on them:
        nan_ones = normed.isnan()
        normed = normed.clone()
        normed[nan_ones] = cls.__DUMMY_VAL
        # New: we might see very large vals here, which may mess up
        # the network.  Let us instead store the squashed normed:
        normed = torch.sigmoid(normed)
        return normed.reshape(-1, 1, 1).to(INPUT_DTYPE)

    def encode_input(self, coarses, *, fines=None, fines2=None, normed=None):
        """Given the key input information, we encode it.

        Dimensions: We expect there to be N deltas and N+1 coarse/fine
        values (since a delta eats the first value).  Then the output
        of this method should be N-1 x 1 x NFEATURES.  Why N-1?
        Because this is the INPUT, and the final coarse/fine/fine2
        value will be used just for targets (which are made
        separately). Unless, of course, we're in genmode, in which
        case we don't save anything for targets.

        Which features? if NO fines: 1-hot of {previous coarse,
        previous delta}. If fines: {previous coarse, previous fine,
        previous delta, current coarse}.  If fines2: {previous coarse,
        previous fine, previous fine2, previous delta, current coarse,
        current fine}

        Arguments:

        coarses: Tensor(NSEQ + 1)

        fines: Tensor(NSEQ + 1), or None

        fines2: Tensor(NSEQ + 1), or None

        normed: Tensor(NSEQ + 1), or None

        Output:

        Each Tensor(NSEQ-1 x NBATCH=1 x NFEATS)  (or NSEQxNBATCHxNFEATS if genmode=True)

        """
        # Here, prefix "p" for previous, "c" for current:
        pcoarses, pnormed, pfines, ccoarses, pfines2, cfines = self.__shift_inputs(
            coarses, fines, fines2, normed)
        feats_list = []
        one_h_pcoarse = self.__one_hot_values(pcoarses, self.ncoarse_bins)
        feats_list.append(one_h_pcoarse)  # always
        if pnormed is not None:
            normed_feats = self.__make_normed_input(pnormed)
            feats_list.append(normed_feats)
        if self.nfine_bins is not None:
            one_h_pfine = self.__one_hot_values(pfines, self.nfine_bins)
            one_h_ccoarse = self.__one_hot_values(ccoarses, self.ncoarse_bins)
            feats_list.insert(1, one_h_pfine)  # prevs to left of covariates
            feats_list.append(one_h_ccoarse)   # currs to right of covariates
            if self.nfine2_bins is not None:
                one_h_pfine2 = self.__one_hot_values(pfines2, self.nfine2_bins)
                one_h_cfine = self.__one_hot_values(cfines, self.nfine_bins)
                feats_list.insert(2, one_h_pfine2)  # prevs just before covariates
                feats_list.append(one_h_cfine)      # currs after everything
        return torch.cat(feats_list, dim=2)

    @staticmethod
    def extract_coarse_extremas(inputs, extremas):
        """The tensor maker knows how the input is divided into coarse and
        extrema features (i.e., just the normed value), so provide
        this method to let them access them.  We use this to enable
        networks to only access the valid part of the input.  Use this
        method when there are no fine or fine2 bins - it's for the
        coarse-only ones, and where we are using extremas (otherwise,
        all the features would be for coarse)

        Arguments:

        inputs: Tensor, NSEQ x NBATCH x NFEATS, the input feature tensor

        extremas: Boolean, if True, we also include features for the
        normed.

        Outputs:

        coarse_inputs: Tensor, NSEQ x NBATCH x NFEATS (it uses all of them)

        extrema_inputs: Tensor, NSEQ x NBATCH x 1 (it uses just the last one)

        """
        coarse_inputs = inputs  # always uses all of them
        if not extremas:
            return coarse_inputs, None
        # Extrema inputs are from the very end:
        nextrema_feats = 1
        extrema_inputs = inputs[:, :, -nextrema_feats:]
        return coarse_inputs, extrema_inputs

    @staticmethod
    def extract_coarse_fine(inputs, ncoarse_bins, extremas):
        """The tensor maker knows how the input is divided into coarse and
        fine features, so provide this method to let them access them.
        We use this to enable networks to only access the valid part
        of the input.  Use this method when there are no *fine2*
        bins. The number nfine_bins is not needed (it's implicit).

        Arguments:

        inputs: Tensor, NSEQ x NBATCH x NFEATS, the input feature tensor

        ncoarse_bins, Int, the number of coarse bins

        extremas: Boolean, if True, we also include features for the
        normed.

        """
        # Recall: prev coarse, prev fine, prev covar, curr coarse --
        # The coarse features are the first 3 groups, the fine are the
        # last three (they overlap).  Each subset uses this many.
        # Logic still works with/without normed features:
        nfeats = inputs.shape[-1]
        nsubset = nfeats - ncoarse_bins
        coarse_inputs = inputs[:, :, :nsubset]
        fine_inputs = inputs[:, :, -nsubset:]
        if not extremas:
            return coarse_inputs, fine_inputs, None
        # The extrema is just the last coarse feature:
        extrema_inputs = inputs[:, :, nsubset-1:nsubset]
        return coarse_inputs, fine_inputs, extrema_inputs

    @staticmethod
    def extract_coarse_fine_fine2(inputs, ncoarse_bins, nfine_bins, extremas):
        """The tensor maker knows how the input is divided into coarse, fine,
        fine2 and extrema features, so we provide this method to let them
        access them.  We use this to enable networks to only access
        the valid part of the input.  Note: The number nfine2_bins is
        not needed (it's implicit).  Use this one when you *do* have
        fine2 bins.

        Arguments:

        inputs: Tensor, NSEQ x NBATCH x NFEATS, the input feature tensor

        ncoarse_bins, Int, the number of coarse bins

        nfine_bins, Int, the number of fine bins

        extremas: Boolean, if True, we also include features for the
        normed.

        Returns:

        coarse_inputs: Tensor, NSEQ x NBATCH x NCOARSE_SUBSET, the coarse-only features.

        fine_inputs: Tensor, NSEQ x NBATCH x NFINE_SUBSET, the fine-only features.

        fine2_inputs: Tensor, NSEQ x NBATCH x NFINE2_SUBSET, the fine2-only features.

        extrema_inputs: Tensor, NSEQ x NBATCH x NEXTREMA_SUBSET, the extrema-only features.

        """
        # *coarse* input is: prev coars, prev fine_, prev fine2, prev covar
        # *fine* input is:               prev fine_, prev fine2, prev covar, curr coars
        # *fine2* input is:                          prev fine2, prev covar, curr coars, curr fine_
        # *total* input is:  prev coars, prev fine_, prev fine2, prev covar, curr coars, curr fine_
        # But note logic still works if no normed features:
        ntotal = inputs.shape[-1]
        coarse_end = ntotal - ncoarse_bins - nfine_bins  # drop last two
        fine_start = ncoarse_bins       # drop first one
        fine_end = ntotal - nfine_bins  # drop last one
        fine2_start = ncoarse_bins + nfine_bins  # drop first two
        coarse_inputs = inputs[:, :, :coarse_end]
        fine_inputs = inputs[:, :, fine_start:fine_end]
        fine2_inputs = inputs[:, :, fine2_start:]
        if not extremas:
            return coarse_inputs, fine_inputs, fine2_inputs, None
        # The extrema is, again, the last coarse feature:
        extrema_inputs = inputs[:, :, coarse_end-1:coarse_end]
        return coarse_inputs, fine_inputs, fine2_inputs, extrema_inputs

    @staticmethod
    def replace_ccoarse(fine_inputs, ccoarse):
        """Background: the inputs can be divided into
        coarse/fine/fine2-specific inputs. The tensor maker knows how
        to replace the ccoarse features in the **fine-specific**
        inputs.  This method does that.

        Arguments:

        fine_inputs: Tensor[1 x NBATCH x NFINE_SUBSET]

        ccoarse: Tensor[NBATCH x NCOARSE]

        """
        # It's the LAST part:
        # *fine* input is:               prev fine_, prev fine2, prev covar, curr coars
        # But note logic still works if no extremas/fine2 features
        ncoarse = ccoarse.shape[-1]
        fine_inputs[0, :, -ncoarse:] = ccoarse
        return fine_inputs

    @staticmethod
    def replace_cfine(fine2_inputs, cfine):
        """Background: the inputs can be divided into
        coarse/fine/fine2-specific inputs. The tensor maker knows how
        to replace the cfine features in the **fine2-specific**
        inputs.  This method does that.

        Arguments:

        fine2_inputs: Tensor[1 x NBATCH x NFINE2_SUBSET]

        cfine: Tensor[NBATCH x NFINE]

        """
        # It's also the last part!
        # *fine2* input is:                          prev fine2, prev covar, curr coars, curr fine_
        # But note logic still works if no extrema features
        nfine = cfine.shape[-1]
        fine2_inputs[0, :, -nfine:] = cfine
        return fine2_inputs

    @staticmethod
    def reset_ccoarse_inputs(inputs, pcoarse, *, pfine=None,
                             pfine2=None, normed=None):
        """Background: we adjust the inputs incrementally over time.  Think of
        this function as the "reset" to be ready for the next
        iteration - basically, make it ready to generate the next
        coarse bins (ccoarse), which is the first step.  It's here in
        the tensor maker because the tensor maker knows how to replace
        the subsets of input features in the overall inputs.  This
        method does that.  The 'p' denotes we are assembling the
        "previously" observed values into this feature vector, to
        generate the next one.

        Arguments:

        inputs: Tensor[1 x NBATCH x NFEATURES]

        pcoarse: Tensor[NBATCH x NCOARSE]

        Optional arguments:

        pfine: Tensor[NBATCH x NFINE], or None if not using

        pfine2: Tensor[NBATCH x NFINE], or None if not using.

        normed: Tensor[NBATCH x 1], or None if not using.

        """
        # *coarse* input is: prev coars, prev fine_, prev fine2, prev covar, UNSEEN PARTS
        # We always replace this one:
        replacement_lst = [pcoarse]
        if pfine is not None:
            replacement_lst.append(pfine)
        if pfine2 is not None:
            replacement_lst.append(pfine2)
        # *coarse* input is: prev coars, prev fine_, prev covar, UNSEEN PARTS
        # prev covar = prev normed
        if normed is not None:
            normed = torch.sigmoid(normed)
            replacement_lst.append(normed)
        if len(replacement_lst) == 1:
            # No need to cat if it's just pcoarse:
            replaced_parts = replacement_lst[0]
        else:
            replaced_parts = torch.cat(replacement_lst, dim=1)
        nfeats = replaced_parts.shape[-1]
        inputs[0, :, :nfeats] = replaced_parts
        return inputs

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

    def __add_masked_extremas(self, normed, stack_list):
        """Add on 'normed' targets for extremas_low and extremas_high, but
        only in those cases where we are in the respective extreme
        bins, otherwise just put IGNORE_INDEX.

        Arguments:

        normed: Tensor[NBATCH], or None if not using.

        stack_list: List[Tensor], the stack list that has the coarse,
        fines, and fines2 targets (if using one or both of the latter
        two, otherwise they are None).  We call it "stack_" because it
        eventually gets stacked via torch.stack().

        Returns:

        nothing, appends to stack_list in place

        """
        # These are the unmasked targets
        normed = normed[2:]
        # Create masks that are True in extreme low/high bins.  Init
        # with coarse situation, then loop through fines (if using):
        coarse_idxs = stack_list[0]
        in_low = coarse_idxs == 0
        in_high = coarse_idxs == self.ncoarse_bins - 1
        nfine12_bins = [self.nfine_bins, self.nfine2_bins]
        for idxs, nbins in zip(stack_list[1:], nfine12_bins):
            if idxs is not None:
                in_low = torch.logical_and(in_low, (idxs == 0))
                in_high = torch.logical_and(in_high, (idxs == (nbins - 1)))
        extremas_low, extremas_high = normed.clone(), normed.clone()
        extremas_low[~in_low] = IGNORE_INDEX
        extremas_high[~in_high] = IGNORE_INDEX
        stack_list.append(extremas_low)
        stack_list.append(extremas_high)

    def encode_target(self, coarses, *, fines=None, fines2=None, normed=None):
        """Given the key OUTPUT information, we encode it.

        Dimensions: Suppose there are N+1 coarse_idxs.  Then the
        output of this method should be N-1 indexes, so a N-1
        dimensional tensor.  Say what?  Yup - you don't include the
        first delta, since there's no input for that (nor the first
        two coarses which make the delta), and actually, your loss
        function just takes the indexes of the true targ_idxs, so
        that's all we need.

        Arguments:

        coarses: Tensor[NSEQ+2]

        fines: Tensor[NSEQ+2], or None if not encoding fines

        fines2: Tensor[NSEQ+2], or None if not encoding fines2

        normed: Tensor[NBATCH], or None if not using.

        Output:

        Tensor(NSEQ x 1 x N), where N is if only doing coarses, and 2
        if doing coarses and fine, and 3 if doing coarses and fine and
        fine2, and 2 more than these if doing extremas in any of those
        cases, where the extrema are always the last 2 targets.

        """
        stack_list = []
        coarse_idxs = coarses[2:]
        nseq = len(coarse_idxs)
        if self.nfine_bins is None:
            # These bin_idxs can map directly to our features.  But we
            # clone first so changes to targets doesn't affect values:
            coarse_idxs = coarse_idxs.clone()
            # Otherwise, no need to clone, because we'll stack them,
            # and that copies into new mem.
        stack_list = [coarse_idxs]
        if self.nfine_bins is not None:
            stack_list.append(fines[2:])
        if self.nfine2_bins is not None:
            stack_list.append(fines2[2:])
        if self.extremas:
            self.__add_masked_extremas(normed, stack_list)
        targets = torch.stack(stack_list, dim=1).reshape(nseq, 1, -1)
        return targets
