"""时序预测残差基线：滚动中位数预测 + 残差标准化。"""
import numpy as np
import pandas as pd

from src import feature_engineering


def residual_scores(payments_df, ref_stats, min_history=3, cap=20.0):
    """
    按 indicator 时序，用"之前所有支付金额"的中位数/MAD 预测下一笔，
    残差 = |x - pred| / pred_scale。冷启动(<min_history 笔)回退到参考期统计。
    """
    scores = np.zeros(len(payments_df))
    # 预计算 indicator 参考期统计兜底（冷启动回退全局）
    fallback_median = feature_engineering.map_with_global(payments_df["indicator_id"], ref_stats, "indicator_id", "median")
    fallback_scale = feature_engineering.map_with_global(payments_df["indicator_id"], ref_stats, "indicator_id", "scale")

    # 按 indicator 分组，维护滚动历史（数据已按时间排序）
    ind_series = payments_df["indicator_id"].values
    amt_series = payments_df["amount_wan"].values
    running = {}   # indicator_id -> list[amount]
    n = len(payments_df)
    for pos in range(n):
        ind = ind_series[pos]
        amt = amt_series[pos]
        hist = running.get(ind, [])
        if len(hist) >= min_history:
            pred = float(np.median(hist))
            mad = float(np.median(np.abs(np.array(hist) - pred))) or 1.0
            resid = abs(amt - pred) / max(mad, 1e-6)
        else:
            # 冷启动：回退到参考期统计（map_with_global 已保证无 NaN）
            pred = fallback_median[pos]
            scale = fallback_scale[pos]
            resid = abs(amt - pred) / max(scale, 1e-6)
        scores[pos] = min(resid, cap)
        running.setdefault(ind, []).append(amt)
    return pd.Series(scores, index=payments_df.index)
