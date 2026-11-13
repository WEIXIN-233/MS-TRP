"""鲁棒统计(MAD)基线：在各实体粒度取 |x-median|/MAD 的最大值。"""
import numpy as np
import pandas as pd

from src import feature_engineering
from baselines._common import _amount


def robust_stats_scores(payments_df, ref_stats):
    """在各实体粒度上取 |x-median|/MAD 的最大值。"""
    amt = _amount(payments_df)
    scores = np.zeros(len(payments_df))
    for col in feature_engineering.STAT_ENTITIES:
        median = feature_engineering.map_with_global(payments_df[col], ref_stats, col, "median")
        scale = feature_engineering.map_with_global(payments_df[col], ref_stats, col, "scale")
        z = np.abs(amt - median) / np.maximum(scale, 1e-6)
        scores = np.maximum(scores, z)
    return pd.Series(scores, index=payments_df.index)
