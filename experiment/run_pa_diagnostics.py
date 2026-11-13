"""
B5 —— 点调整（Point Adjustment）对抗性诊断（diagnostics_pa.csv）
==============================================================
指导书 §8.3.1：--no_pa 为正式模式；--pa_diagnostic 仅执行对抗性诊断。这里产出
f1_raw（冻结阈值下点级 F1，无 PA）与 f1_pa_diag（PA 后 F1）及 pa_inflation=差值。
diagnostics_pa.csv 仅诊断，不用于主排名或模型选择。

PA 定义（诊断口径）：对每个真事件，若其任一成员被报警，则把该事件全部成员视作命中。
schema：model,data_seed,f1_raw,f1_pa_diag,pa_inflation

用法：python -m experiment.run_pa_diagnostics [--data-seed 10] [--out result_summary/diagnostics_pa.csv]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import _export_common as ec
from src import config, data, split, metrics


def _f1(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(config.RESULT_DIR, "diagnostics_pa.csv"))
    args = ap.parse_args()

    payments, labels, events, point_labels = ec.load_standard()
    train_idx, val_idx, test_idx = split.time_forward_split(payments)
    y_test = point_labels["is_anomaly"].astype(int).values[test_idx]
    apply_test = payments["apply_id"].astype(str).iloc[test_idx].values

    pred = ec.load_predictions(args.data_seed)
    pred = pred[["apply_id", "s_final"]].copy()
    pred["apply_id"] = pred["apply_id"].astype(str)
    s_map = pred.set_index("apply_id")["s_final"].to_dict()
    s_test = np.asarray([s_map.get(a, 0.0) for a in apply_test], dtype=float)
    thr = ec.load_thresholds(args.data_seed)["thr_global"]

    pred_flag = (s_test > thr).astype(int)
    f1_raw = _f1(y_test, pred_flag)

    # PA：事件任一成员命中 → 全成员命中
    test_set = set(apply_test)
    flagged_set = set(apply_test[pred_flag == 1])
    pred_pa = pred_flag.copy()
    for _, ev in events.iterrows():
        members = [str(a) for a in ev["apply_ids"] if str(a) in test_set]
        if not members:
            continue
        if any(m in flagged_set for m in members):
            for m in members:
                idx = np.where(apply_test == m)[0]
                if len(idx):
                    pred_pa[idx[0]] = 1
    f1_pa = _f1(y_test, pred_pa)

    rows = [{
        "model": "MS-TRP",
        "data_seed": args.data_seed if args.data_seed is not None else "main",
        "f1_raw": round(f1_raw, 6),
        "f1_pa_diag": round(f1_pa, 6),
        "pa_inflation": round(f1_pa - f1_raw, 6),
    }]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
