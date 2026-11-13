"""
B5/B7 —— 事件级预测 / 匹配 / 参数（event_predictions.csv + event_matches.csv + config/event_params.json）
========================================================================================================
指导书 §8.3.1：Type-agnostic Event-F1，以主要对象、连续告警时间间隔≤Δ_event 合并预测事件；
与真事件按主要对象一致 + 时间交叠做一对一最大匹配。Δ_event 与阈值均验证集冻结，测试集不重调。

event_params.json：object rule、merge_gap、threshold、tie handling、version。

用法：python -m experiment.run_events [--data-seed 10]
      [--out-pred result_raw/event_predictions.csv --out-match result_raw/event_matches.csv
       --out-params config/event_params.json]
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _export_common as ec
from src import config, data, split
from src import event_eval

DELTA_EVENT = event_eval.DELTA_EVENT_DEFAULT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--out-pred", default=os.path.join(config.RESULT_DIR, "event_predictions.csv"))
    ap.add_argument("--out-match", default=os.path.join(config.RESULT_DIR, "event_matches.csv"))
    ap.add_argument("--out-params", default=os.path.join(config.PROJECT_DIR, "config", "event_params.json"))
    args = ap.parse_args()

    payments, labels, events, point_labels = ec.load_standard()
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    thr = ec.load_thresholds(args.data_seed)["thr_global"]

    pred = ec.load_predictions(args.data_seed)
    pred["apply_id"] = pred["apply_id"].astype(str)
    s_map = dict(zip(pred["apply_id"], pred["s_final"]))

    apply_test = payments["apply_id"].astype(str).iloc[test_idx].values
    test_set = set(apply_test)
    test_events = events[events["apply_ids"].apply(lambda ids: any(str(a) in test_set for a in ids))].reset_index(drop=True)

    # 报警支付（test，冻结阈值）
    test_idx_arr = np.asarray(test_idx, dtype=int)
    flagged_idx = test_idx_arr[np.asarray([s_map.get(a, 0.0) for a in apply_test]) > thr]
    flagged = payments.iloc[flagged_idx].copy()
    flagged["s_final"] = [s_map[a] for a in flagged["apply_id"].astype(str)]

    # 预测事件
    true_ev = event_eval.build_true_events(test_events)
    if len(true_ev) and len(flagged):
        alerts = event_eval.build_payment_alerts(flagged)
        pred_ev = event_eval.cluster_alerts(alerts, delta_event=DELTA_EVENT)
        match = event_eval.match_events(true_ev, pred_ev)
    else:
        pred_ev = pd.DataFrame(columns=["entity_key", "start", "end", "n_alerts", "apply_ids"])
        match = {"tp": 0, "fn": int(len(true_ev)), "fp": 0,
                 "event_recall": 0.0, "event_precision": 0.0, "event_f1": 0.0, "pairs": []}

    # event_predictions.csv
    pred_rows = []
    for i, (_, pe) in enumerate(pred_ev.iterrows()):
        pred_rows.append({
            "pred_event_id": f"PE{i:06d}",
            "entity_key": pe["entity_key"],
            "start": str(pd.Timestamp(pe["start"]).date()),
            "end": str(pd.Timestamp(pe["end"]).date()),
            "n_alerts": int(pe["n_alerts"]),
            "apply_ids": ",".join(str(a) for a in pe["apply_ids"]),
        })
    pd.DataFrame(pred_rows).to_csv(args.out_pred, index=False, encoding="utf-8-sig")

    # event_matches.csv（一对一匹配明细）
    true_rows = []
    for i, (_, te) in enumerate(true_ev.iterrows()):
        true_rows.append({"true_event_idx": i, "entity_key": te["entity_key"],
                          "start": str(pd.Timestamp(te["start"]).date()),
                          "end": str(pd.Timestamp(te["end"]).date())})
    true_df = pd.DataFrame(true_rows)
    match_rows = []
    for ti, pi in match["pairs"]:
        match_rows.append({
            "true_event_idx": ti, "pred_event_idx": pi,
            "entity_key": true_df.iloc[ti]["entity_key"],
            "tp": 1,
        })
    pd.DataFrame(match_rows).to_csv(args.out_match, index=False, encoding="utf-8-sig")

    # event_params.json
    params = {
        "object_rule": "entity_id 一致 + 时间交叠",
        "merge_gap": DELTA_EVENT,
        "threshold": thr,
        "tie_handling": "score > threshold 报警",
        "version": "v1",
        "type_agnostic": True,
        "one_to_one_matching": True,
        "data_seed": args.data_seed if args.data_seed is not None else "main",
        "event_f1": match["event_f1"],
        "tp": match["tp"], "fp": match["fp"], "fn": match["fn"],
    }
    os.makedirs(os.path.dirname(args.out_params), exist_ok=True)
    with open(args.out_params, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print(f"[B5] event_predictions.csv：{len(pred_rows)} 个预测事件")
    print(f"[B5] event_matches.csv：{len(match_rows)} 对匹配")
    print(f"[B5] event_params.json：event_f1={match['event_f1']:.4f} (TP={match['tp']}, FP={match['fp']}, FN={match['fn']})")


if __name__ == "__main__":
    main()
