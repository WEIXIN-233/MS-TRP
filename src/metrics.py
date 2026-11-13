"""
MS-TRP 检测框架 —— 评估指标
========================================
阈值无关：AUPRC、ROC-AUC
阈值相关：点级 P/R/F1、事件级检出率/精确率/F1、各类召回率、每千笔误报数

事件级口径（在代码注释中明确）：
  - 事件 = 一条标签(anchor_apply_id ∪ affected_apply_ids)对应的支付集合
  - 事件检出率(event_recall) = 至少命中一个成员的事件占比
  - 事件精确率(event_precision) = 命中某事件的报警支付 / 全部报警支付
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config


def auprc(y_true, scores):
    return average_precision_score(y_true, scores)


def roc_auc(y_true, scores):
    return roc_auc_score(y_true, scores)


def normalized_ap_excess(y_true, scores):
    """
    归一化 AP 超额（P0-4 改正：原误称 VUS-PR 的指标）。
    (AUPRC - π) / (1 - π)，π = 正样本率。相对随机基线的精度增益，random→0、perfect→1。
    """
    y = np.asarray(y_true, dtype=float)
    pi = y.mean()
    if pi >= 1.0:
        return 0.0
    ap = average_precision_score(y_true, scores)
    return max((ap - pi) / (1.0 - pi), 0.0)


def metrics_at_topk(y_true, scores, k_ratio):
    """
    报警预算工作点（Top-K% 回顾性预算，P0-1 改正：不再称为"固定误报率"）：
    把分数最高的 k_ratio 比例判为异常，返回 P/R/F1 与预算内命中数。
    """
    y = np.asarray(y_true, dtype=int)
    sc = np.asarray(scores, dtype=float)
    k = max(1, int(round(len(y) * k_ratio)))
    top_idx = np.argsort(-sc)[:k]
    pred = np.zeros(len(y), dtype=int)
    pred[top_idx] = 1
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"k": k, "precision": p, "recall": r, "f1": f1}


def per_type_event_recall(events, apply_id_to_score, thr):
    """每类异常的事件级召回率（至少命中一个成员的事件占比）。"""
    flagged = {aid for aid, s in apply_id_to_score.items() if s > thr}
    result = {}
    for at in config.ANOMALY_TYPES:
        sub = events[events["anomaly_type"] == at]
        caught = sum(1 for _, ev in sub.iterrows() if ev["apply_ids"] & flagged)
        result[at] = caught / len(sub) if len(sub) > 0 else 0.0
    return result


def best_threshold(y_true, scores):
    """在给定分数上扫描阈值，返回使点级 F1 最大的阈值与对应 P/R/F1。
    用排序+前缀和实现 O(n log n)，避免逐阈值全量扫描。"""
    y = np.asarray(y_true, dtype=int)
    sc = np.asarray(scores, dtype=float)
    n = len(y)
    n_pos = int(y.sum())
    if n_pos == 0:
        return (0.0, 0.0, 0.0, 0.0)

    order = np.argsort(sc, kind="mergesort")
    sc_sorted = sc[order]
    y_sorted = y[order]
    cum_pos = np.cumsum(y_sorted)          # cum_pos[i] = 前 i+1 个(按分数升序)里的正样本数

    best = (0.0, 0.0, 0.0, 0.0)
    for t in np.unique(sc_sorted):
        idx = int(np.searchsorted(sc_sorted, t, side="right"))  # 分数>t 的起始位置
        n_pred = n - idx
        if n_pred == 0:
            continue
        pos_le = int(cum_pos[idx - 1]) if idx > 0 else 0        # 分数<=t 的正样本
        tp = n_pos - pos_le
        fp = n_pred - tp
        fn = n_pos - tp
        p = tp / n_pred
        r = tp / n_pos
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        if f1 > best[3]:
            best = (float(t), p, r, f1)
    return best


def point_metrics(y_true, scores, thr):
    """点级 P/R/F1 + TP/FP/FN。"""
    pred = (np.asarray(scores, dtype=float) > thr).astype(int)
    y = np.asarray(y_true, dtype=int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def fpr(n_fp, n_normal):
    """FPR = FP / (FP + TN) = FP / N_normal。"""
    return n_fp / n_normal if n_normal > 0 else 0.0


def far_per_1000(n_fp, n_normal):
    """指导书 §4.2 权威口径：FAR/1000 = FP / N_normal × 1000（N_normal = 正常支付数）。"""
    return n_fp / n_normal * 1000 if n_normal > 0 else 0.0


def fp_per_1000_all(n_total, n_fp):
    """每千笔支付（含异常）的误报数——旁证口径，非指导书 FAR/1000。"""
    return n_fp / n_total * 1000 if n_total > 0 else 0.0


def evaluate_point(y_true, scores, threshold):
    """
    统一点级评估入口（P0-4 口径，主实验/基线/消融共用）：
      阈值无关：AUPRC / ROC-AUC / normalized_ap_excess
      阈值相关：P/R/F1、TP/FP/FN、FPR、FAR/1000（正常支付分母）、fp_per_1000_all（旁证）
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pm = point_metrics(y, s, threshold)
    n_total = len(y)
    n_normal = n_total - int(y.sum())
    return {
        "auprc": average_precision_score(y, s),
        "auc": roc_auc_score(y, s),
        "normalized_ap_excess": normalized_ap_excess(y, s),
        "precision": pm["precision"], "recall": pm["recall"], "f1": pm["f1"],
        "tp": pm["tp"], "fp": pm["fp"], "fn": pm["fn"],
        "fpr": fpr(pm["fp"], n_normal),
        "far_per_1000": far_per_1000(pm["fp"], n_normal),
        "fp_per_1000_all": fp_per_1000_all(n_total, pm["fp"]),
    }


def event_metrics(events, apply_id_to_score, thr):
    """
    事件级指标。events: DataFrame(apply_ids=set)。apply_id_to_score: dict。
    """
    flagged = {aid for aid, s in apply_id_to_score.items() if s > thr}
    n_events = len(events)
    caught = 0
    for _, ev in events.iterrows():
        if ev["apply_ids"] & flagged:
            caught += 1
    event_recall = caught / n_events if n_events > 0 else 0.0

    # 事件精确率 = 命中事件的报警支付 / 全部报警支付
    all_event_ids = set()
    for _, ev in events.iterrows():
        all_event_ids |= ev["apply_ids"]
    hit = flagged & all_event_ids
    event_precision = len(hit) / len(flagged) if flagged else 0.0
    f1 = 2 * event_recall * event_precision / (event_recall + event_precision) \
        if event_recall + event_precision > 0 else 0.0
    return {"event_recall": event_recall, "event_precision": event_precision,
            "event_f1": f1, "caught": caught, "n_events": n_events}


def per_class_recall(point_labels, apply_id_to_score, thr):
    """
    每类异常的点级召回率：命中该类的报警支付 / 该类受影响支付。
    point_labels: DataFrame(apply_id, anomaly_types)。
    """
    flagged = {aid for aid, s in apply_id_to_score.items() if s > thr}
    result = {}
    for at in config.ANOMALY_TYPES:
        members = set(point_labels[point_labels["anomaly_types"].apply(
            lambda lst: at in lst)]["apply_id"].astype(str))
        hit = members & flagged
        result[at] = len(hit) / len(members) if members else 0.0
    return result


def flag_payments(scores_series, apply_ids, thr):
    """把分数序列与 apply_id 对齐成 {apply_id: score} 字典。"""
    return {str(aid): float(s) for aid, s in zip(apply_ids, scores_series)}


# ── 三族指标：点级 / 区间级(VUS-PR) / 排序级(NDCG@K) ──────────────
# 指导书 B5/Q3 统一 schema；metrics_family.csv 由本映射导出。
METRIC_FAMILIES = {
    "point": ["point_auprc", "point_auc", "normalized_ap_excess",
              "point_precision", "point_recall", "point_f1", "far_per_1000"],
    "interval": ["vus_pr"],
    "ranking": ["ndcg_at_k"],
}


def vus_pr(y_true, scores, timestamps):
    """
    区间级 VUS-PR（体积积分）——真正的时间感知指标，非 normalized_ap_excess。

    把测试集按自然月切成 K 个区间，每个区间内算一个点级 AUPRC（PR 曲线下的面积），
    再按区间样本量加权求和，得到「PR × 时间」曲面下的归一化体积：
        VUS-PR = Σ_k w_k · AUPRC_k ，w_k = n_k / Σ n_k
    含义：异常检测质量在时间轴上的稳定性/一致性。只在少数月份能检出的方法，
    即使全局 AUPRC 高，VUS-PR 也会被拉低；逐月都稳定检出则趋近全局 AUPRC。
    区间内无正样本、全正样本、或分数退化为常数时该区间跳过（AUPRC 无定义）。
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    ts = pd.to_datetime(timestamps)
    periods = ts.dt.to_period("M")
    vol = 0.0
    w_total = 0.0
    n_valid = 0
    for _, idx in pd.Series(np.arange(len(y))).groupby(periods.values).groups.items():
        idx = np.asarray(idx, dtype=int)
        yk, sk = y[idx], s[idx]
        n_pos, n_neg = int(yk.sum()), int((1 - yk).sum())
        if n_pos == 0 or n_neg == 0 or len(np.unique(sk)) < 2:
            continue  # 该区间 AUPRC 无定义
        w = len(idx)
        vol += w * average_precision_score(yk, sk)
        w_total += w
        n_valid += 1
    if w_total <= 0:
        return 0.0
    return float(vol / w_total)


def ndcg_at_k(y_true, scores, k=None):
    """
    排序级 NDCG@K（二值相关度：异常=1、正常=0）。
    DCG@K = Σ_{i=1..K} rel_i / log2(i+1)；IDCG 取理想排序（正样本全排最前）。
    K 缺省 = 测试集正样本数 n_pos（衡量「能否把全部异常都顶到最前」）。
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.0
    if k is None or k <= 0 or k > len(y):
        k = n_pos
    if k <= 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    top = y[order[:k]]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(top * discounts))
    ideal_k = min(k, n_pos)
    idcg = float(np.sum(discounts[:ideal_k]))
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)
