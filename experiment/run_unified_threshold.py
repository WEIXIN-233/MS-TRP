# -*- coding: utf-8 -*-
"""审稿意见第3条（优先级2）：统一验证集阈值协议下的检测器比较。

所有方法（含 MS-TRP）使用同一条阈值规则：
  在验证集正常支付上取分数 0.99 分位（目标误报率 0.01），冻结后到测试集只评一次。
这样精确率/召回率/FAR/Event-F1 的差别只来自分数，不来自各自阈值策略。

输出 result_raw/unified_threshold.csv：
  method, thr_unified, auprc, precision, recall, f1, far_1000, event_recall, event_f1

用法：python -m experiment.run_unified_threshold
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data, metrics, split  # noqa: E402
from src import feature_engineering as features  # noqa: E402
from experiment.run_baselines import BASELINE_FUNCS  # noqa: E402

ALPHA = 0.01


def freeze_threshold(scores, val_idx, y_all):
    """验证集正常样本的 (1-alpha) 分位分。"""
    val = np.asarray(scores)[val_idx]
    norm = val[np.asarray(y_all)[val_idx] == 0]
    return float(np.quantile(norm, 1.0 - ALPHA))


def eval_at(scores, test_idx, y_all, events, test_aids, thr):
    s = np.asarray(scores)[test_idx].astype(float)
    y = np.asarray(y_all)[test_idx].astype(int)
    ev = metrics.evaluate_point(y, s, thr)
    score_map = {str(a): float(v) for a, v in zip(test_aids, s)}
    test_events = events[events["apply_ids"].apply(
        lambda ids: bool(ids & set(test_aids)))].reset_index(drop=True)
    em = metrics.event_metrics(test_events, score_map, thr)
    return {
        "auprc": metrics.auprc(y, s),
        "precision": ev["precision"], "recall": ev["recall"], "f1": ev["f1"],
        "far_1000": ev["far_per_1000"],
        "event_recall": em["event_recall"], "event_f1": em["event_f1"],
    }


def main():
    payments = data.load_payments()
    labels = data.load_labels()
    events = data.build_events(labels)
    point_labels = data.build_point_labels(payments, events)
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    y_all = point_labels["is_anomaly"].astype(int).values
    test_aids = point_labels.loc[test_idx, "apply_id"].astype(str).values

    ref_stats = features.compute_reference_stats(payments, train_idx)

    rows = []

    # ── 基线 ──
    for name, func in BASELINE_FUNCS.items():
        scores = func(payments, ref_stats, train_idx).values.astype(float)
        thr = freeze_threshold(scores, val_idx, y_all)
        rows.append({"method": name, "thr_unified": round(thr, 6),
                     **{k: round(v, 6) for k, v in
                        eval_at(scores, test_idx, y_all, events, test_aids, thr).items()}})
        print(rows[-1], flush=True)

    # ── MS-TRP：复用主运行逐样本分数 ──
    pred = pd.read_csv(os.path.join(config.RESULT_DIR, "run_main", "predictions.csv"))
    pred["apply_id"] = pred["apply_id"].astype(str)
    full = pred.set_index("apply_id").reindex(
        point_labels["apply_id"].astype(str).values)
    assert full["s_final"].notna().all(), "主运行 predictions 与当前数据不对齐"
    s_ms = full["s_final"].values.astype(float)
    thr = freeze_threshold(s_ms, val_idx, y_all)
    rows.append({"method": "MS-TRP", "thr_unified": round(thr, 6),
                 **{k: round(v, 6) for k, v in
                    eval_at(s_ms, test_idx, y_all, events, test_aids, thr).items()}})
    print(rows[-1], flush=True)

    out = os.path.join(config.RESULT_DIR, "unified_threshold.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print("written:", out)


if __name__ == "__main__":
    main()
