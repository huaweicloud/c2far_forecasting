"""Utility that tracks how sequence-based parameters (like csize,
gsize, lagfeat_period) change after aggregation.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/ml/multivariate/multi_freq_mapper.py


class MultiFreqMapper():
    @staticmethod
    def divide_and_check(value, agg_amt):
        """Helper to divide the value by the given agg_amt, and check for
        various problems.

        """
        if value is None:
            return None
        elif value % agg_amt != 0:
            msg = f"Cannot aggregate {value} evenly with {agg_amt}."
            raise RuntimeError(msg)
        return value // agg_amt
