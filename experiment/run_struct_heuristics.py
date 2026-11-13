"""
B5 —— 零训练结构启发式指标（heuristic_structure_metrics.csv）
=============================================================
指导书 §8.3.3：Shared-Payee Count 与 Shared-Account Concentration 两种零训练结构基线，
只用评分时点前合法关系；在 ENTITY_AGGREGATION 上的分组 Recall 与 Recall@FAR 须和 Graph/PPR
成对比较。schema：model,data_seed,window_days,coverage,entity_event_recall,recall_at_far5,paired_effect。

口径：
  · coverage = 测试集上可计算该结构分数的支付占比（payee/account 缺失或历史不足为 0，仍可算）。
  · entity_event_recall = ENTITY_AGGREGATION 测试事件在冻结阈值下的至少命中一成员召回。
  · recall_at_far5 = 固定 5% 误报率（阈值=正常样本分数的 95 分位）下的点级召回。
  · paired_effect = 该启发式 entity_event_recall − MS-TRP entity_event_recall（成对差）。

用法：python -m experiment.run_struct_heuristics [--data-seed 10]
      [--out result_summary/heuristic_structure_metrics.csv]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _export_common as ec
from src import config, data, split, metrics, calibration
import baselines as blm


def recall_at_far(y, s, target_far=0.05):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    normal = s[y == 0]
    pos = s[y == 1]
    if len(normal) == 0 or len(pos) == 0:
        return np.nan
    thr = np.quantile(normal, 1.0 - target_far)
    return float((pos > thr).mean())


def entity_event_recall(events, scores, apply_ids, thr):
    """ENTITY_AGGREGATION 测试事件召回（至少命中一成员）。"""
    ea = events[events["anomaly_type"] == "ENTITY_AGGREGATION"]
    score_map = dict(zip(apply_ids, scores))
    caught = 0
    for _, ev in ea.iterrows():
        if any(score_map.get(str(a), 0.0) > thr for a in ev["apply_ids"] if str(a) in score_map):
            caught += 1
    return caught / len(ea) if len(ea) else np.nan, len(ea)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(config.RESULT_DIR, "heuristic_structure_metrics.csv"))
    args = ap.parse_args()

    payments, labels, events, point_labels = ec.load_standard()
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    y_all = point_labels["is_anomaly"].astype(int).values
    ref_stats = None

    funcs = {
        "shared_payee_count": blm.shared_payee_count_scores,
        "shared_account_concentration": blm.shared_account_concentration_scores,
    }
    apply_ids = payments["apply_id"].astype(str).tolist()
    test_set = set(apply_ids[i] for i in test_idx)
    test_events = events[events["apply_ids"].apply(lambda ids: any(str(a) in test_set for a in ids))]

    # MS-TRP 参照
    pred = ec.load_predictions(args.data_seed)
    pred = pred[["apply_id", "s_final"]].copy()
    pred["apply_id"] = pred["apply_id"].astype(str)
    s_ms = pred.set_index("apply_id")["s_final"].reindex(apply_ids).values
    thr_ms = ec.load_thresholds(args.data_seed)["thr_global"]
    ms_rec, _ = entity_event_recall(test_events, s_ms, apply_ids, thr_ms)

    rows = []
    for name, fn in funcs.items():
        scores = fn(payments, ref_stats, train_idx).values
        s_val = scores[val_idx]
        y_val = y_all[val_idx]
        thr = calibration.calibrate_target_fpr(s_val, y_val, alpha_fpr=calibration.ALPHA_FPR_DEFAULT)
        y_test = y_all[test_idx]
        s_test = scores[test_idx]
        # coverage：score 可计算（非 NaN 且非全零因缺 payee/account）
        coverage = float(np.isfinite(scores[test_idx]).mean())
        rec, n_ea = entity_event_recall(test_events, scores, apply_ids, thr)
        r5 = recall_at_far(y_test, s_test, 0.05)
        rows.append({
            "model": name,
            "data_seed": args.data_seed if args.data_seed is not None else "main",
            "window_days": 30,
            "coverage": round(coverage, 6),
            "entity_event_recall": round(rec, 6) if np.isfinite(rec) else np.nan,
            "recall_at_far5": round(r5, 6) if np.isfinite(r5) else np.nan,
            "paired_effect": round((rec - ms_rec), 6) if np.isfinite(rec) else np.nan,
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"[B5] MS-TRP entity_event_recall(EAG) = {ms_rec:.4f}；ENTITY_AGGREGATION 测试事件 = 参考")


if __name__ == "__main__":
    main()
