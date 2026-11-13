"""
MS-TRP 检测框架 —— 特征工程
========================================
核心原则：历史统计只在"参考期"(train 段)上计算，对 val/test 天然无泄漏。
实体粒度：indicator(序列级) / unit(单位级) / econ_class(经济分类) / pay_type(支付类型)
特征：金额原始值、对数金额、相对各粒度历史中位数/均值的放大倍数、时间特征。
"""
import numpy as np
import pandas as pd

# 历史统计涉及的实体列
STAT_ENTITIES = ["indicator_id", "unit_name", "econ_class_code", "pay_type_code"]


def compute_reference_stats(payments_df: pd.DataFrame, ref_idx) -> dict:
    """
    在参考期(ref_idx)上计算各实体的金额统计：median/mean/mad/std。
    返回 {entity_col: DataFrame(index=实体值, cols=[median,mean,mad,std,count])}
    """
    ref = payments_df.loc[ref_idx]
    stats = {}
    for col in STAT_ENTITIES:
        g = ref.groupby(col)["amount_wan"].agg(
            median="median", mean="mean", mad=_mad, std="std", count="size")
        # MAD 可能为 0，用 std 兜底
        g["scale"] = np.where(g["mad"] > 0, g["mad"], np.where(g["std"] > 0, g["std"], 1.0))
        stats[col] = g
    # 全局兜底（冷启动实体回退到全局统计）
    amt = ref["amount_wan"]
    stats["__global__"] = {
        "median": amt.median(), "mean": amt.mean(),
        "mad": _mad(amt), "scale": max(_mad(amt), amt.std(), 1e-6),
    }
    return stats


def map_with_global(keys, ref_stats, col, stat_name):
    """
    按实体列映射参考期统计，冷启动(未在参考期出现的实体)回退到全局值。
    返回与 keys 等长的 numpy 数组，无 NaN。
    """
    st = ref_stats[col]
    g = ref_stats["__global__"][stat_name]
    mapped = keys.map(st[stat_name])
    return mapped.fillna(g).values


def _mad(s: pd.Series) -> float:
    """中位数绝对偏差 MAD = median(|x - median|)。"""
    m = s.median()
    return (s - m).abs().median()


def _ratio(a, b):
    """a/b，避免除零与极端值：b<=0 时返回 NaN。"""
    return np.where(b > 0, a / np.maximum(b, 1e-6), np.nan)


def build_feature_matrix(payments_df: pd.DataFrame, ref_stats: dict) -> pd.DataFrame:
    """
    基于参考期统计，为全量支付构建特征矩阵（对 val/test 无泄漏）。
    返回 DataFrame，列名以 f_ 前缀。
    """
    df = payments_df.copy()
    F = pd.DataFrame(index=df.index)

    # 原始金额 + 对数
    F["f_amount"] = df["amount_wan"]
    F["f_log_amount"] = np.log1p(df["amount_wan"])

    # 相对各粒度历史统计
    for col in STAT_ENTITIES:
        st = ref_stats[col]
        median = df[col].map(st["median"])
        mean = df[col].map(st["mean"])
        scale = df[col].map(st["scale"])
        cnt = df[col].map(st["count"])
        F[f"f_{col}_median_ratio"] = _ratio(df["amount_wan"], median)
        F[f"f_{col}_mean_ratio"] = _ratio(df["amount_wan"], mean)
        F[f"f_{col}_zscore"] = _ratio(df["amount_wan"] - median, scale)
        F[f"f_{col}_hist_count"] = cnt.fillna(0)

    # 时间特征
    F["f_month"] = df["month"]
    F["f_is_december"] = df["is_december"]
    F["f_year"] = df["year"]

    return F


def _safe_log(x):
    return np.log1p(np.maximum(x, 0))
