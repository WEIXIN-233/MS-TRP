"""
MS-TRP Step 7 —— 节点级可靠度门控融合
========================================
时序可靠度 r_ts：有效历史量、1−基线波动（历史越充分越稳定越可信）。
图可靠度 r_g：有效邻居数、关系结构信号强度（账户一致性/实体聚集越强越可信）。
门控权重：[g_ts, g_g] = softmax([r_ts/τ, r_g/τ])，最终风险 s = g_ts·s_ts + g_g·s_g。
τ 只在验证集选择；本文件给出默认值 τ=1.0。
"""
import numpy as np
import pandas as pd

TAU_DEFAULT = 1.0
HIST_SAT = 20.0      # 历史笔数达到该值视为充分
DEG_SAT = 5.0        # 有效邻居数达到该值视为充分
EPS = 1e-6


def compute_temporal_reliability(feat):
    """r_ts：历史充分度 与 历史稳定性 的均值，均归一化到 [0,1]。"""
    hist_adequacy = np.clip(feat["hist_count"] / HIST_SAT, 0.0, 1.0)
    cv = np.abs(feat["hist_std"]) / (feat["hist_mean"].abs() + EPS)
    stability = 1.0 / (1.0 + np.nan_to_num(cv, nan=1.0))
    r_ts = 0.5 * hist_adequacy + 0.5 * stability
    return np.clip(r_ts, 0.0, 1.0)


def compute_graph_reliability(degree, s_struct):
    """r_g：有效邻居数（归一化）与 结构信号强度 的均值。"""
    deg_norm = np.clip(degree / DEG_SAT, 0.0, 1.0)
    r_g = 0.5 * deg_norm + 0.5 * s_struct
    return np.clip(r_g, 0.0, 1.0)


def gate_fusion(s_ts, s_g, r_ts, r_g, tau=TAU_DEFAULT):
    """
    门控融合。输入均为等长 array。
    返回 (s, g_ts, g_g)。
    """
    e_ts = np.exp(np.asarray(r_ts, dtype=float) / tau)
    e_g = np.exp(np.asarray(r_g, dtype=float) / tau)
    z = e_ts + e_g + EPS
    g_ts = e_ts / z
    g_g = e_g / z
    s = g_ts * np.asarray(s_ts, dtype=float) + g_g * np.asarray(s_g, dtype=float)
    return s, g_ts, g_g
