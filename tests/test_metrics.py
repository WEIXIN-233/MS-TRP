# -*- coding: utf-8 -*-
"""评价指标口径单元测试（P0-4 验收工具）。

指导书 §4.2 权威定义：
  P=TP/(TP+FP)  R=TP/(TP+FN)  F1=2PR/(P+R)
  FPR=FP/(FP+TN)=FP/N_normal
  FAR/1000 = FP / N_normal × 1000   （N_normal = 正常支付数 = FP+TN）
  AP 用 average_precision_score（非插值）
"""
import numpy as np

from src import metrics


def test_point_metrics_a05():
    """学长验收用例 A05：TP=2, FP=1, FN=2, TN=5。"""
    y = np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    thr = 0.5
    m = metrics.point_metrics(y, s, thr)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 2)
    assert abs(m["precision"] - 2 / 3) < 1e-9
    assert abs(m["recall"] - 0.5) < 1e-9
    assert abs(m["f1"] - 4 / 7) < 1e-9


def test_point_metrics_20_samples():
    """P0-4 验收：20 条小数据，已知 TP=3, FP=2, FN=4, TN=11。"""
    y = np.array([1] * 7 + [0] * 13)
    s = np.array([0.9, 0.8, 0.7,              # 3 TP
                  0.1, 0.1, 0.1, 0.1,         # 4 FN
                  0.75, 0.6,                  # 2 FP
                  0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 11 TN
    thr = 0.5
    m = metrics.point_metrics(y, s, thr)
    assert (m["tp"], m["fp"], m["fn"]) == (3, 2, 4)
    p = 3 / 5
    r = 3 / 7
    assert abs(m["precision"] - p) < 1e-9
    assert abs(m["recall"] - r) < 1e-9
    assert abs(m["f1"] - 2 * p * r / (p + r)) < 1e-9
    # N_normal = TN + FP = 13
    assert abs(metrics.fpr(2, 13) - 2 / 13) < 1e-9
    assert abs(metrics.far_per_1000(2, 13) - 2 / 13 * 1000) < 1e-9


def test_far_per_1000_target():
    """固化指导书 FAR/1000 权威公式（分母=正常支付数）。"""
    fp, tn = 1, 5
    n_normal = fp + tn
    assert abs(metrics.far_per_1000(fp, n_normal) - 1 / 6 * 1000) < 1e-6
    assert abs(metrics.fpr(fp, n_normal) - 1 / 6) < 1e-9


def test_normalized_ap_excess():
    """归一化 AP 超额 = max((AP−π)/(1−π), 0)，random→0。"""
    y = np.array([0, 0, 0, 1, 0])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.4])   # 正样本分数最高 → AP=1
    ap = metrics.auprc(y, s)
    pi = y.mean()
    assert abs(metrics.normalized_ap_excess(y, s) - (ap - pi) / (1 - pi)) < 1e-9


def test_metrics_at_topk_consistency():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.4, 0.5, 0.6, 0.0])
    m = metrics.metrics_at_topk(y, s, 0.3)   # top-3 = 0.9/0.8/0.7 → 全为真
    assert m["precision"] == 1.0
    assert abs(m["recall"] - 1.0) < 1e-9
    assert abs(m["f1"] - 1.0) < 1e-9


def test_auprc_roc_auc_sanity():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.3, 0.1, 0.7, 0.2, 0.6, 0.4, 0.5, 0.0])
    assert 0.0 <= metrics.auprc(y, s) <= 1.0
    assert 0.0 <= metrics.roc_auc(y, s) <= 1.0


def test_evaluate_point_single_entry():
    """统一入口 evaluate_point 与各分量一致。"""
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.4, 0.5, 0.6, 0.0])
    ev = metrics.evaluate_point(y, s, 0.5)
    pm = metrics.point_metrics(y, s, 0.5)
    assert ev["precision"] == pm["precision"]
    assert ev["recall"] == pm["recall"]
    assert ev["f1"] == pm["f1"]
    assert abs(ev["far_per_1000"] - metrics.far_per_1000(pm["fp"], 7)) < 1e-9
