# -*- coding: utf-8 -*-
"""
2025+ 新基线公共数据准备 + 指标计算（与 metrics_main.csv 同口径）。
=================================================================
复用主流程的 load_standard / time_forward_split / build_feature_matrix，
产出 prep.npz（特征 X、标签 y、train/val/test 索引），
并提供 report() 用主评测器（metrics.evaluate_point / event_strict_f1）算
AUPRC / P / R / F1 / FAR 及事件 F1。

用法（在项目根目录）：
    python -m baselines_2025._prep              # 生成 prep.npz 并打印摘要
    from baselines_2025._prep import report      # 各方法脚本复用
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, metrics, calibration, feature_engineering, split
from experiment import _export_common as ec
from baselines._common import _impute

HERE = os.path.dirname(os.path.abspath(__file__))
PREP_PATH = os.path.join(HERE, "prep.npz")


def build_prep():
    """构建并缓存 prep.npz，返回 (X, y, train_idx, val_idx, test_idx, payments, events)。"""
    if os.path.exists(PREP_PATH):
        z = np.load(PREP_PATH, allow_pickle=True)
        payments = pd.read_pickle(os.path.join(HERE, "prep_payments.pkl"))
        events = pd.read_pickle(os.path.join(HERE, "prep_events.pkl"))
        return (z["X"], z["y"], z["train_idx"], z["val_idx"], z["test_idx"], payments, events)

    payments, labels, events, point_labels = ec.load_standard()
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    ref_stats = feature_engineering.compute_reference_stats(payments, train_idx)
    X = feature_engineering.build_feature_matrix(payments, ref_stats)
    # 与主基线（LOF/IF 等）一致：NaN 用列中位数填充、inf 置 0（冷启动实体回退）
    X = _impute(X)
    y = point_labels["is_anomaly"].astype(int).values

    np.savez(PREP_PATH, X=X.values.astype(np.float32), y=y.astype(np.int32),
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    payments.to_pickle(os.path.join(HERE, "prep_payments.pkl"))
    events.to_pickle(os.path.join(HERE, "prep_events.pkl"))
    return X.values.astype(np.float32), y, train_idx, val_idx, test_idx, payments, events


def report(name, scores_full, payments=None):
    """
    给定全量分数（与 payments 等长 array），打印一行与 metrics_main 同口径的指标。
    冻结阈值 = 验证集 target-FPR 校准（与基线一致：alpha_fpr 默认 0.01）。
    """
    X, y, train_idx, val_idx, test_idx, pay, events = build_prep()
    if payments is None:
        payments = pay
    y = y.astype(int)
    s = np.asarray(scores_full, dtype=float)
    s_val, s_test = s[val_idx], s[test_idx]
    y_val, y_test = y[val_idx], y[test_idx]

    thr = calibration.calibrate_target_fpr(s_val, y_val,
                                           alpha_fpr=calibration.ALPHA_FPR_DEFAULT)
    pe = metrics.evaluate_point(y_test, s_test, thr)
    ev = ec.event_strict_f1(payments, events, test_idx=test_idx, scores=s, thr=thr)

    row = {
        "model": name,
        "payment_auprc": round(pe["auprc"], 6),
        "precision": round(pe["precision"], 6),
        "recall": round(pe["recall"], 6),
        "f1": round(pe["f1"], 6),
        "far_1000": round(pe["far_per_1000"], 4),
        "event_f1": round(ev["event_f1"], 6),
    }
    print(f"[{name}] " + "  ".join(f"{k}={v}" for k, v in row.items()))
    return row


if __name__ == "__main__":
    X, y, train_idx, val_idx, test_idx, payments, events = build_prep()
    print(f"X.shape={X.shape}  n={len(y)}  "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}  "
          f"prevalence(test)={y[test_idx].mean():.6f}")
    print("特征列数:", X.shape[1], "（FoMo-0D/OutFormer 需补零到 100）")
