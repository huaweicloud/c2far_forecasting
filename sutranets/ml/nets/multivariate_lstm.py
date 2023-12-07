"""Long-short-term memory neural network that processes sequences from
various sub-series.  Internally, it feeds each sequence to a separate
FCLSTM (and also can be Joint/Triple), which generates the logits at
that level.  It then re-assembles the output logits here into a
concatenated output across all sequences.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/nets/multivariate_lstm.py
import os
import pathlib
import torch
from sutranets.ml.nets.multi_lstm import MultiLSTM


class MultivariateLSTM(MultiLSTM):
    def forward(self, minibatch):
        """Pass in a minibatch of training examples, then slice into the
        inputs for each frequency and run the corresponding forward
        pass on each.

        """
        all_outputs = []
        input_lst = minibatch.get_lst_of_tensors()
        for model, inputs in zip(self.models, input_lst):
            outputs = model.forward(inputs)
            all_outputs.append(outputs)
        concat_outs = torch.stack(all_outputs)
        return concat_outs

    @staticmethod
    def __yield_model_paths(outpath, extension, nsub_series, *, mkdir):
        """Helper to handle creating the model paths for us."""
        if mkdir:
            pathlib.Path(outpath).mkdir(parents=True, exist_ok=False)
        for i in range(nsub_series):
            model_base = f"submodel.{i}{extension}"
            model_path = os.path.join(outpath, model_base)
            yield model_path

    def _get_model_paths(self, outpath, extension):
        """Call class-specific method for getting model paths."""
        nsub_series = len(self.models)
        return self.__yield_model_paths(
            outpath, extension, nsub_series, mkdir=True)

    @classmethod
    def create_from_path(cls, path, submodel_cls, nsub_series, *, device=None):
        """Factory method that returns an instance of this class, given the
        model at the current filename.

        """
        models = []
        for model_path in cls.__yield_model_paths(path, ".pt", nsub_series, mkdir=False):
            model = submodel_cls.create_from_path(model_path, device=device)
            models.append(model)
        new_model = cls(models)
        return new_model
