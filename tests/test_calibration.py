# -*- coding: utf-8 -*-
"""P0-1 阈值校准单元测试（calibrate_target_fpr / 报警预算改名）。"""
import numpy as np

from src import calibration


def test_calibrate_target_fpr_basic():
    """目标 FPR 阈值 = 验证集正常样本的 1−alpha_fpr 分位。"""
    val_scores = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.90])
    val_labels = np.array([0, 0, 0, 0, 0, 1])
    tau = calibration.calibrate_target_fpr(val_scores, val_labels, alpha_fpr=0.2)
    normal = val_scores[val_labels == 0]
    assert abs(tau - float(np.quantile(normal, 0.8))) < 1e-12


def test_calibrate_target_fpr_excludes_positives():
    """只统计正常样本，异常样本不进入分位。"""
    val_scores = np.array([0.1, 0.2, 0.3, 0.9, 0.99])
    val_labels = np.array([0, 0, 0, 1, 1])
    tau = calibration.calibrate_target_fpr(val_scores, val_labels, alpha_fpr=0.5)
    assert abs(tau - 0.2) < 1e-12   # quantile([0.1,0.2,0.3], 0.5) = 0.2


def test_alert_budget_ratio_rename_alias():
    """P0-1 改名：ALPHA_HIGH 保留为 ALERT_BUDGET_RATIO 的兼容别名。"""
    assert calibration.ALPHA_HIGH == calibration.ALERT_BUDGET_RATIO
