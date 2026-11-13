"""
MS-TRP Step 8 —— 分层阈值校准
========================================
在验证集上按"单位类型 × 项目类别 × 执行阶段"分组确定阈值；组内样本不足时
向全局阈值收缩：τ'_c = α_c·τ_c + (1−α_c)·τ_global，α_c = n_c/(n_c+γ)。
测试阶段阈值冻结，直接用验证集阈值判定 Normal/Warning/HighRisk。
"""
import numpy as np
import pandas as pd

from src import metrics

GAMMA_DEFAULT = 30.0      # 平滑常数（验证集可调）
ALPHA_LOW = 0.05          # 低位阈值分位（Warning）
ALERT_BUDGET_RATIO = 0.01 # 报警预算比例（P0-1 改名，原 ALPHA_HIGH）：Top-quantile 回顾性预算
ALPHA_HIGH = ALERT_BUDGET_RATIO   # 兼容旧名（弃用）
ALPHA_FPR_DEFAULT = 0.01  # 目标 FPR 校准的 alpha_fpr（在验证集正常样本上冻结）


def _group_threshold(scores, labels, quantile):
    """给定组内分数与标签，返回使 top-quantile 判为异常的分位数阈值。"""
    sc = np.asarray(scores, dtype=float)
    if len(sc) == 0:
        return 0.0
    # 报警预算工作点（Top-quantile 回顾性预算，P0-1 改正：不是固定误报率）
    k = max(1, int(round(len(sc) * quantile)))
    return float(np.sort(sc)[-k]) if k <= len(sc) else float(sc.min())


def calibrate_hierarchical(val_scores, val_labels, val_groups, gamma=GAMMA_DEFAULT,
                           quantile=ALPHA_HIGH):
    """
    返回 {group_key: threshold}，含全局兜底 "__global__"。
    val_scores/val_labels/val_groups 均与验证集样本对齐。
    """
    sc = np.asarray(val_scores, dtype=float)
    lab = np.asarray(val_labels, dtype=int)
    grp = np.asarray(val_groups, dtype=object)

    tau_global = _group_threshold(sc, lab, quantile)

    # 组内统计
    df = pd.DataFrame({"g": grp, "s": sc, "y": lab})
    counts = df.groupby("g").size()
    thresholds = {}
    for g, sub in df.groupby("g"):
        n_c = len(sub)
        tau_c = _group_threshold(sub["s"].values, sub["y"].values, quantile)
        alpha = n_c / (n_c + gamma)
        thresholds[str(g)] = float(alpha * tau_c + (1.0 - alpha) * tau_global)

    thresholds["__global__"] = float(tau_global)
    return thresholds


def calibrate_target_fpr(val_scores, val_labels, alpha_fpr=ALPHA_FPR_DEFAULT):
    """
    目标 FPR 校准（P0-1 新增）：在验证集正常样本上取 1−alpha_fpr 分位作为阈值。
      tau = quantile(val_scores[y==0], 1 − alpha_fpr)
    验证集与测试集共用同一冻结阈值（验收判据④）；可线上复现。
    """
    sc = np.asarray(val_scores, dtype=float)
    lab = np.asarray(val_labels, dtype=int)
    normal = sc[lab == 0]
    if len(normal) == 0:
        return float(np.quantile(sc, 1.0 - alpha_fpr)) if len(sc) else 0.0
    return float(np.quantile(normal, 1.0 - alpha_fpr))


def apply_thresholds(scores, groups, thresholds):
    """按组应用阈值，返回预测等级 0=Normal / 1=Warning / 2=HighRisk（此处为单阈值二值）。"""
    out = np.zeros(len(scores), dtype=int)
    for i, (s, g) in enumerate(zip(scores, groups)):
        tau = thresholds.get(str(g), thresholds.get("__global__", 0.5))
        if s > tau:
            out[i] = 1
    return out
