"""固定规则基线：绝对大额 / 相对历史中位数、均值的放大倍数。"""
import numpy as np
import pandas as pd

from src import config, feature_engineering
from baselines._common import _amount


def fixed_rules_scores(payments_df, ref_stats):
    """连续分数 = 各条规则"违反倍数"的最大值。"""
    amt = _amount(payments_df)
    scores = np.zeros(len(payments_df))

    # 规则1：绝对大额
    scores = np.maximum(scores, amt / config.FIXED_RULES["amount_abs_threshold_wan"])

    # 规则2/3：相对 indicator 历史中位数/均值
    ind_median = feature_engineering.map_with_global(payments_df["indicator_id"], ref_stats, "indicator_id", "median")
    ind_mean = feature_engineering.map_with_global(payments_df["indicator_id"], ref_stats, "indicator_id", "mean")
    scores = np.maximum(scores,
                        amt / np.maximum(ind_median * config.FIXED_RULES["amount_median_mult"], 1e-6))
    scores = np.maximum(scores,
                        amt / np.maximum(ind_mean * config.FIXED_RULES["amount_mean_mult"], 1e-6))
    return pd.Series(scores, index=payments_df.index)
