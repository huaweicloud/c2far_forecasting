"""Aggregate a higher-frequency series into a lower-frequency one.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/dataprep/aggregate_seqs.py
from copy import copy
from enum import Enum
import numpy as np
from sutranets.dataprep.data_utils import DataUtils


class AggFunction(Enum):
    """Types of supported aggregation functions."""
    SUBSERIES = "subseries"


def collect_and_agg(meta, vals, agg_amt, agg_csize, agg_gsize, agg_func, agg_offset=None):
    """Aggregate the given time series by the given amount. Then collect
    the final csize+gsize+2 that you need to make tensors.

    Arguments:

    meta: list[String], our normal meta-data, from one line in a trace
    file.

    vals: list[Float], normal set of time series values.

    agg_csize/agg_gsize, ints, the intendend csize and gsize at the
    given aggregation level.

    agg_func: AggFunction, how we wish to aggregate the data.

    agg_offset: Int, required if using AggFunction.SUBSERIES, as we
    take the series at the given offset.

    """
    targ_amt = agg_csize + agg_gsize + 2
    pre_agg_amt = targ_amt * agg_amt
    targ_vals = vals[-pre_agg_amt:]
    if agg_amt == 1:
        return meta, targ_vals
    meta, agged = agg_series(meta, targ_vals, agg_amt, agg_func, agg_offset)
    return meta, agged


def agg_series(meta, vals, agg_period, agg_func, agg_offset=None):
    """Aggregate a single time series using the given agg_period.

    Arguments:

    meta: list[String], our normal meta-data, from one line in a trace
    file.

    vals: list[Float], normal set of time series values.

    agg_period: Int, the period over which we aggregate.

    agg_func: AggFunction, how we wish to aggregate the data.

    agg_offset: Int, required if using AggFunction.SUBSERIES, as we
    take the series at the given offset.

    """
    new_meta = copy(meta)
    samp_per = DataUtils.get_sampling_period_s(new_meta)
    new_samp_per = samp_per * agg_period
    DataUtils.set_sampling_period_s(new_meta, new_samp_per)
    if type(vals) is list:
        vals = np.array(vals)
    vals = vals.reshape(-1, agg_period)
    if agg_func == AggFunction.SUBSERIES:
        reverse_offset = (agg_period - 1) - agg_offset
        new_last_s = DataUtils.get_last_s(meta) - reverse_offset * samp_per
        DataUtils.set_last_s(new_meta, new_last_s)
    vals = vals[:, agg_offset]
    return new_meta, vals
