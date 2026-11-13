"""
B5 —— 结构启发式 vs PPR 成对比较（struct_vs_ppr_paired.csv）
============================================================
指导书 §8.3.3：Shared-Payee/Account 计数、Simple PPR、Full MS-TRP 在 ENTITY_AGGREGATION
上的分组 Recall 与 Recall@FAR 成对比较。schema：
  model,data_seed,payment_auprc,entity_event_recall,shared_payee_recall,shared_account_recall,recall_at_far5

· entity_event_recall = 全部 ENTITY_AGGREGATION 测试事件召回（冻结阈值）。
· shared_payee_recall / shared_account_recall = 按 Shared Payee / Shared Account 子类型分组的召回。
· 不预设"结构启发式必然弱于 PPR"；照实报告。

用法：python -m experiment.run_struct_ppr_paired [--data-seed 10]
      [--out result_raw/struct_vs_ppr_paired.csv]
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


def event_recall(events, subtype, scores, apply_ids, thr):
    """subtype: None=全部 EA；'PAYEE'/'ACCOUNT' 按 target_entity_type 分组。"""
    sub = events[events["anomaly_type"] == "ENTITY_AGGREGATION"]
    if subtype:
        sub = sub[sub["target_entity_type"] == subtype]
    score_map = dict(zip(apply_ids, scores))
    caught = 0
    for _, ev in sub.iterrows():
        if any(score_map.get(str(a), 0.0) > thr for a in ev["apply_ids"] if str(a) in score_map):
            caught += 1
    return caught / len(sub) if len(sub) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(config.RESULT_DIR, "struct_vs_ppr_paired.csv"))
    args = ap.parse_args()

    payments, labels, events, point_labels = ec.load_standard()
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    y_all = point_labels["is_anomaly"].astype(int).values
    apply_ids = payments["apply_id"].astype(str).tolist()
    test_set = set(apply_ids[i] for i in test_idx)
    test_events = events[events["apply_ids"].apply(lambda ids: any(str(a) in test_set for a in ids))]

    pred = ec.load_predictions(args.data_seed)
    pred = pred[["apply_id", "s_final"]].copy()
    pred["apply_id"] = pred["apply_id"].astype(str)
    s_ms = pred.set_index("apply_id")["s_final"].reindex(apply_ids).values
    thr_ms = ec.load_thresholds(args.data_seed)["thr_global"]

    ref_stats = None
    models = {
        "shared_payee_count": blm.shared_payee_count_scores(payments, ref_stats, train_idx).values,
        "shared_account_concentration": blm.shared_account_concentration_scores(payments, ref_stats, train_idx).values,
        "simple_ppr": blm.simple_ppr_scores(payments).values,
        "MS-TRP": s_ms,
    }

    rows = []
    for name, scores in models.items():
        if name == "MS-TRP":
            thr = thr_ms
        else:
            thr = calibration.calibrate_target_fpr(scores[val_idx], y_all[val_idx],
                                                    alpha_fpr=calibration.ALPHA_FPR_DEFAULT)
        y_test = y_all[test_idx]
        rows.append({
            "model": name,
            "data_seed": args.data_seed if args.data_seed is not None else "main",
            "payment_auprc": round(metrics.auprc(y_test, scores[test_idx]), 6),
            "entity_event_recall": round(event_recall(test_events, None, scores, apply_ids, thr), 6),
            "shared_payee_recall": round(event_recall(test_events, "PAYEE", scores, apply_ids, thr), 6),
            "shared_account_recall": round(event_recall(test_events, "ACCOUNT", scores, apply_ids, thr), 6),
            "recall_at_far5": round(recall_at_far(y_test, scores[test_idx]), 6),
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
