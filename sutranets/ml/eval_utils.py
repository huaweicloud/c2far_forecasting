"""Utilities used for evaluation.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/eval_utils.py
import logging
import os
from sutranets.dataprep.cutoffs.utils import load_cutoffs, LoadCutoffsError
from sutranets.file_utils import setup_outdir
from sutranets.ml.batch_predictor_mgr import BatchPredictorMgr
from sutranets.ml.constants import IntraBinDecoder
from sutranets.ml.evaluators.generation_evaluator import GenerationEvaluator
from sutranets.ml.multivariate.hf_multiv_batch_predictor_mgr import HFMultivBatchPredictorMgr
from sutranets.utils import init_console_logger, init_file_logger
DEFAULT_LOGLEVS = [("sutranets", logging.INFO)]
LOG_SHORT_FN = "log.txt"
logger = logging.getLogger("sutranets.ml.eval_utils")


class EvalUtils():
    """Static utilities used specifically in evaluation."""
    __CUTOFF_TOLERANCE = 1e-4

    @staticmethod
    def _make_log_dir_plus_fn(args):
        """Helper to create the logging (output) dir, create the path to the
        logging file, and return this path.

        """
        subdir_parts = []
        subdir_parts.append(os.path.basename(args.test_offs))
        outdir = os.path.join(args.logging_dir, *subdir_parts)
        target_dir = setup_outdir(outdir)
        log_fn = os.path.join(target_dir, LOG_SHORT_FN)
        return log_fn

    @classmethod
    def setup_logging(cls, args, logger_levels=None):
        """Helper function to set up all the logging."""
        if logger_levels is None:
            logger_levels = DEFAULT_LOGLEVS
        init_console_logger(logger_levels)
        if args.logging_dir is not None:
            log_fn = cls._make_log_dir_plus_fn(args)
            init_file_logger(logger_levels, log_fn)
        logger.info("ARGS: %s", str(vars(args)))

    @staticmethod
    def __validate_create_gen_eval(lstm_eval, coarse_cutoffs, args):
        key_arg_names = ["nsamples", "confidence", "nstride", "lstm_evaluator"]
        key_arg_values = [args.nsamples, args.confidence_pct, args.nstride, lstm_eval]
        for arg_nm, arg in zip(key_arg_names, key_arg_values):
            if arg is None:
                raise RuntimeError(f"Required argument '{arg_nm}' is unexpectedly None")
        key_arg_names = ["gen_bdecoder", "coarse_cutoffs"]
        key_arg_values = [args.gen_bdecoder, coarse_cutoffs]
        for arg_nm, arg in zip(key_arg_names, key_arg_values):
            if arg is None:
                raise RuntimeError(f"'{arg_nm}' required for non-continuous eval")
        if not isinstance(args.gen_bdecoder, IntraBinDecoder):
            raise RuntimeError(f"gen_bdecoder {args.gen_bdecoder} is not an IntraBinDecoder enum")

    @classmethod
    def create_gen_eval(cls, lstm_eval, coarse_cutoffs, args):
        """Helper to return an LSTM-based multi-step generation evaluator.
        Validates some of the needed arguments.

        """
        cls.__validate_create_gen_eval(lstm_eval, coarse_cutoffs, args)
        if args.nsub_series is not None:
            bpred_mgr = HFMultivBatchPredictorMgr.create(
                lstm_eval, coarse_cutoffs, args, args.gen_bdecoder)
        else:
            bpred_mgr = BatchPredictorMgr.create(
                lstm_eval, coarse_cutoffs, args, args.gen_bdecoder)
        gen_eval = GenerationEvaluator(bpred_mgr, args.device)
        return gen_eval

    @classmethod
    def create_multi_evaluators(cls, lstm_eval, coarse_cutoffs, args):
        systems = []
        if lstm_eval is not None:
            gen_eval = cls.create_gen_eval(lstm_eval, coarse_cutoffs,
                                           args)
            systems.append(gen_eval)
        return systems

    @classmethod
    def validate_cutoffs(cls, model_path, coarse_cutoffs):
        if coarse_cutoffs is None:
            return
        model_dir = os.path.dirname(model_path)
        try:
            saved_cutoffs = load_cutoffs(model_dir)
        except LoadCutoffsError:
            logger.error("No cutoffs file in %s, skipping check.", model_dir)
            return
        if len(saved_cutoffs) != len(coarse_cutoffs):
            msg = f"Saved cutoffs different length from created ones: {coarse_cutoffs}"
            raise RuntimeError(msg)
        for cut1, cut2 in zip(saved_cutoffs, coarse_cutoffs):
            if abs(cut1 - cut2) > cls.__CUTOFF_TOLERANCE:
                msg = f"Saved cutoffs different than created ones: {coarse_cutoffs}"
                raise RuntimeError(msg)
